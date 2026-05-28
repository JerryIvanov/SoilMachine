/*
================================================================================
                          Layermap Data Structure
================================================================================

Concept: Heightmap representet as a grid. Each grid element points to the "top"
element of a double-linked-list of RLE components which contain the layer meta data.
These components are additionally memory pooled for efficiency.

Utilizes a vertexpool to avoid remeshing when updating heightmap information

A layermap section has a height, a porosity and a water fraction.
The idea is that if the water fraction is 1, then all pores are filled.
This tells me the overall saturation of water inside.
Layermap has a height, a porosity and a

Air e.g. has porosity 1

*/

//#define SOILMACHINE_MASK

#include "include/FastNoiseLite.h"

#include <glm/glm.hpp>
#include <atomic>
#include <mutex>
#include <vector>
#include <cstdint>
using namespace glm;
using namespace std;

/*
================================================================================
                  RLE Linked-List Element and Memory Pool
================================================================================
*/

#include "surface.h"

struct sec {

sec* next = NULL;     //Element Above
sec* prev = NULL;     //Element Below

SurfType type = soilmap["Air"];   //Type of Surface Element
double size = 0.0f;               //Run-Length of Element
double floor = 0.0f;              //Cumulative Height at Bottom
double saturation = 0.0f;         //Saturation with Water

sec(){}
sec(double s, SurfType t){
  size = s;
  type = t;
}

void reset(){
  next = NULL;
  prev = NULL;
  type = soilmap["Air"];
  size = 0.0f;
  floor = 0.0f;
  saturation = 0.0f;
}

};

// Thread-local кэш sec-элементов: каждый поток держит мелкий локальный запас.
// get/unget обращаются к глобальному пулу только при пустом/переполненном кэше (batch).
// Это сводит contention на pool_mutex к минимуму: 1 lock на batch вместо 1 lock на элемент.
// inline thread_local — C++17, одна переменная на поток через все TU.
inline thread_local std::vector<sec*> tl_sec_cache;
static constexpr int TL_SEC_CACHE_MAX  = 64; // flush-порог
static constexpr int TL_SEC_BATCH_SIZE = 32; // размер batch get/flush

class secpool {
public:

int size = 0;
sec* start = NULL;
std::vector<sec*> free_global;     // глобальный пул
mutable std::mutex pool_mutex;     // защищает free_global

secpool(){}
secpool(const int N){ reserve(N); }
~secpool(){ delete[] start; }

void reserve(const int N){
  start = new sec[N];
  size  = N;
  free_global.reserve(N);
  for(int i = 0; i < N; i++) free_global.push_back(&start[i]);
}

template<typename... Args>
sec* get(Args && ...args){
  // Быстрый путь: берём из thread-local кэша (без лока)
  if(!tl_sec_cache.empty()){
    sec* E = tl_sec_cache.back();
    tl_sec_cache.pop_back();
    try{ new (E)sec(forward<Args>(args)...); }
    catch(...){ throw; }
    return E;
  }
  // Медленный путь: batch-пополнение из глобального пула
  {
    std::lock_guard<std::mutex> lk(pool_mutex);
    int n = (int)std::min((size_t)TL_SEC_BATCH_SIZE, free_global.size());
    if(n == 0){ cout<<"Memory Pool Out-Of-Elements"<<endl; return NULL; }
    for(int i = 0; i < n; i++){
      tl_sec_cache.push_back(free_global.back());
      free_global.pop_back();
    }
  }
  sec* E = tl_sec_cache.back();
  tl_sec_cache.pop_back();
  try{ new (E)sec(forward<Args>(args)...); }
  catch(...){ throw; }
  return E;
}

void unget(sec* E){
  if(!E) return;
  E->reset();
  tl_sec_cache.push_back(E);
  // Flush в глобальный пул когда кэш переполнен
  if((int)tl_sec_cache.size() > TL_SEC_CACHE_MAX){
    std::lock_guard<std::mutex> lk(pool_mutex);
    for(int i = 0; i < TL_SEC_BATCH_SIZE; i++){
      free_global.push_back(tl_sec_cache.back());
      tl_sec_cache.pop_back();
    }
  }
}

void reset(){
  // Однопоточно: сбрасываем локальный кэш, перестраиваем глобальный пул
  tl_sec_cache.clear();
  free_global.clear();
  free_global.reserve(size);
  for(int i = 0; i < size; i++) free_global.push_back(&start[i]);
}

};

/*
================================================================================
                      Queriable Layermap Datastructure
================================================================================
*/

class Layermap {

private:

sec** dat = NULL;                         //Raw Data Grid

public:

ivec2 dim;                                //Size
secpool pool;                             //Data Pool

// Thread safety: atomic_flag spinlock (~3-5нс) вместо recursive_mutex (~50нс).
// add() / remove() разбиты на locked wrapper + add_internal() / remove_internal()
// чтобы избежать рекурсивного захвата (add_internal рекурсирует без лока).
static const int CELL_LOCK_STRIPES = 4096;
mutable std::atomic_flag cell_flags[CELL_LOCK_STRIPES]; // C++20: default = false

static int cell_stripe(ivec2 pos) {
    uint64_t h = (uint64_t)(uint32_t)pos.x * 2654435761ULL
               ^ (uint64_t)(uint32_t)pos.y * 1234567891ULL;
    return (int)(h % CELL_LOCK_STRIPES);
}
void acquire_cell(int s) const {
    while(cell_flags[s].test_and_set(std::memory_order_acquire));
}
void release_cell(int s) const {
    cell_flags[s].clear(std::memory_order_release);
}

//Queries
double height(ivec2);                     //Query Height at Discrete Position
double height(vec2);                      //Query Height at Position (Bilinear Interpolation)
vec3 normal(ivec2);                       //Normal Vector at Position
vec3 normal(vec2);                        //Normal Vector at Position (Bilinear Interpolation)
vec3 normal(ivec2, Vertexpool<Vertex>&);  //Normal Vector at Position (Read from Vertexpool)
vec3 normal(vec2, Vertexpool<Vertex>&);   //Normal Vector at Position (Read from Vertexpool)
SurfType surface(ivec2);                  //Surface Type at Position

//Modifiers
void add(ivec2, sec*);                    //Add Layer at Position (acquires cell spinlock)
void add_internal(ivec2, sec*);           //Add Layer — no lock, called recursively
double remove(ivec2, double);             //Remove Layer at Position (acquires cell spinlock)
double remove_internal(ivec2, double);    //Remove Layer — no lock
sec* top(ivec2 pos){                      //Top Element at Position
  return dat[pos.x*dim.y+pos.y];
}

//Meshing / Visualization
uint* section = NULL;                     //Vertexpool Section Pointer
void meshpool(Vertexpool<Vertex>&);       //Mesh based on Vertexpool
void update(ivec2, Vertexpool<Vertex>&);  //Update Vertexpool at Position (No Remesh)
void update(Vertexpool<Vertex>&);         //Update Vertexpool at Position (No Remesh)
void slice(Vertexpool<Vertex>&, double);         //Update Vertexpool at Position (No Remesh)

public:

void initialize(int SEED, ivec2 _dim){

  dim = _dim;

  //Important so Re-Callable

  pool.reset();

  if(dat != NULL) delete[] dat;
  dat = new sec*[dim.x*dim.y];      //Array of Section Pointers

  for(int i = 0; i < dim.x; i++)
  for(int j = 0; j < dim.y; j++)
    dat[i*dim.y+j] = NULL;

  //Fill 'er up

  const int MAXSEED = 10000;
  for(size_t l = 0; l < layers.size(); l++){

    const float f = (float)l/(float)layers.size();
    const int Z = SEED + f*MAXSEED;
    layers[l].init();

    //Add a first layer!
    for(int i = 0; i < dim.x; i++){
    for(int j = 0; j < dim.y; j++){

      double h = layers[l].get(vec3(i, j, Z%MAXSEED)/vec3(dim.x, dim.y, 1));
      add(ivec2(i, j), pool.get(h, layers[l].type));

    }}

  }

  // Mask the World

  #ifdef SOILMACHINE_MASK

  layers[0].scale = 1.0f;
  layers[0].bias = 0.5f;

  for(int i = 0; i < dim.x; i++)
  for(int j = 0; j < dim.y; j++){

    double maxh = layers[0].get(vec3(i, j, SEED%MAXSEED)/vec3(dim.x, dim.y, MAXSEED));
    float diff = remove(ivec2(i, j), height(ivec2(i, j)) - maxh);
    while(diff > 0) diff = remove(ivec2(i, j), height(ivec2(i, j)) - maxh);

  }

  #endif

}

//Constructors
Layermap(int SEED, ivec2 _dim){
  pool.reserve(POOLSIZE);                //Some permissible amount of RAM later...
  initialize(SEED, _dim);
}

Layermap(int SEED, ivec2 _dim, Vertexpool<Vertex>& vertexpool):Layermap(SEED, _dim){
  meshpool(vertexpool);
}

};

// Locked wrapper — захватывает spinlock ячейки, затем вызывает add_internal
void Layermap::add(ivec2 pos, sec* E){
  int s = cell_stripe(pos);
  acquire_cell(s);
  add_internal(pos, E);
  release_cell(s);
}

// Реализация без лока — рекурсивные вызовы тоже идут через add_internal
void Layermap::add_internal(ivec2 pos, sec* E){

  //Non-Element: Don't Add
  if(E == NULL)
    return;

  //Negative Size Element: Don't Add
  if(E->size <= 0){
    pool.unget(E);
    return;
  }

  //Valid Element, Empty Spot: Set Top Directly
  if(dat[pos.x*dim.y+pos.y] == NULL){
    dat[pos.x*dim.y+pos.y] = E;
    return;
  }

  //Valid Element, Previous Type Identical: Elongate
  if(dat[pos.x*dim.y+pos.y]->type == E->type){
    dat[pos.x*dim.y+pos.y]->size += E->size;
    pool.unget(E);
    return;
  }

  //Basically: A position Swap

  //Add to Water, but not equal to water
  if(dat[pos.x*dim.y+pos.y]->type == soilmap["Air"]){ //Switch with Water

    //Remove Top Element (Water)
    sec* top = dat[pos.x*dim.y+pos.y];
    dat[pos.x*dim.y+pos.y] = top->prev;

    //Add this Element (без повторного захвата лока)
    add_internal(pos, E);

    //Add Water Back In
    add_internal(pos, top);

    return;
  }

  //Add Element
  dat[pos.x*dim.y+pos.y]->next = E;
  E->prev = dat[pos.x*dim.y+pos.y];
  E->floor = height(pos);
  dat[pos.x*dim.y+pos.y] = E;

}

// Locked wrapper
double Layermap::remove(ivec2 pos, double h){
  int s = cell_stripe(pos);
  acquire_cell(s);
  double result = remove_internal(pos, h);
  release_cell(s);
  return result;
}

// Реализация без лока
double Layermap::remove_internal(ivec2 pos, double h){

  //No Element to Remove
  if(dat[pos.x*dim.y+pos.y] == NULL)
    return 0.0;

  //Element Needs Removal
  if(dat[pos.x*dim.y+pos.y]->size <= 0.0){
    sec* E = dat[pos.x*dim.y+pos.y];
    dat[pos.x*dim.y+pos.y] = E->prev; //May be NULL
    pool.unget(E);
    return 0.0;
  }

  //No Removal Necessary (Note: Zero Height Elements Removed)
  if(h <= 0.0)
    return 0.0;

  double diff = h - dat[pos.x*dim.y+pos.y]->size;
  dat[pos.x*dim.y+pos.y]->size -= h;

  if(diff >= 0.0){
    sec* E = dat[pos.x*dim.y+pos.y];
    dat[pos.x*dim.y+pos.y] = E->prev; //May be NULL
    pool.unget(E);
    return diff;
  }
  else return 0.0;

}

vec3 Layermap::normal(ivec2 pos){

  vec3 n = vec3(0);
  vec3 p = vec3(pos.x, SCALE*height(pos), pos.y);
  int k = 0;

  if(pos.x > 0 && pos.y > 0){
    vec3 b = vec3(pos.x-1, SCALE*height(pos-ivec2(1,0)), pos.y);
    vec3 c = vec3(pos.x, SCALE*height(pos-ivec2(0,1)), pos.y-1);
    n += cross(c-p, b-p);
    k++;
  }

  if(pos.x > 0 && pos.y < dim.y - 1){
    vec3 b = vec3(pos.x-1, SCALE*height(pos-ivec2(1,0)), pos.y);
    vec3 c = vec3(pos.x, SCALE*height(pos+ivec2(0,1)), pos.y+1);
    n -= cross(c-p, b-p);
    k++;
  }

  if(pos.x < dim.x-1 && pos.y > 0){
    vec3 b = vec3(pos.x+1, SCALE*height(pos+ivec2(1,0)), pos.y);
    vec3 c = vec3(pos.x, SCALE*height(pos-ivec2(0,1)), pos.y-1);
    n -= cross(c-p, b-p);
    k++;
  }

  if(pos.x < dim.x-1 && pos.y < dim.y-1){
    vec3 b = vec3(pos.x+1, SCALE*height(pos+ivec2(1,0)), pos.y);
    vec3 c = vec3(pos.x, SCALE*height(pos+ivec2(0,1)), pos.y+1);
    n += cross(c-p, b-p);
    k++;
  }

  return normalize(n/(float)k);

}

vec3 Layermap::normal(vec2 pos){

  vec3 n = vec3(0);
  ivec2 p = floor(pos);
  vec2 w = fract(pos);

  n += (1.0f-w.x)*(1.0f-w.y)*normal(p);
  n += (1.0f-w.x)*w.y*normal(p+ivec2(1,0));
  n += w.x*(1.0f-w.y)*normal(p+ivec2(0,1));
  n += w.x*w.y*normal(p+ivec2(1,1));

  return n;

}

vec3 Layermap::normal(ivec2 pos, Vertexpool<Vertex>& vertexpool){
  return normal(pos);
  //Vertex* v = vertexpool.get(section, pos.x*dim.y+pos.y);
  //return vec3(v->normal[0], v->normal[1], v->normal[2]);
}

vec3 Layermap::normal(vec2 pos, Vertexpool<Vertex>& vertexpool){

  vec3 n = vec3(0);
  ivec2 p = floor(pos);
  vec2 w = fract(pos);

  n += (1.0f-w.x)*(1.0f-w.y)*normal(p, vertexpool);
  n += (1.0f-w.x)*w.y*normal(p+ivec2(1,0), vertexpool);
  n += w.x*(1.0f-w.y)*normal(p+ivec2(0,1), vertexpool);
  n += w.x*w.y*normal(p+ivec2(1,1), vertexpool);

  return n;

}

//Queries

SurfType Layermap::surface(ivec2 pos){
  sec* s = dat[pos.x*dim.y+pos.y]; // single load — atomic на x86-64
  if(s == NULL) return 0;
  return s->type;
}

double Layermap::height(ivec2 pos){
  sec* s = dat[pos.x*dim.y+pos.y]; // single load — избегаем double-read nullptr
  if(s == NULL) return 0.0;
  return (s->floor + s->size);
}

double Layermap::height(vec2 pos){

  double h = 0.0f;
  ivec2 p = floor(pos);
  vec2 w = fract(pos);

  h += (1.0-w.x)*(1.0-w.y)*height(p);
  h += (1.0-w.x)*w.y*height(p+ivec2(1,0));
  h += w.x*(1.0-w.y)*height(p+ivec2(0,1));
  h += w.x*w.y*height(p+ivec2(1,1));
  return h;

}

// Meshing, Updating

void Layermap::meshpool(Vertexpool<Vertex>& vertexpool){

  if(section != NULL){
    vertexpool.unsection(section);
    vertexpool.indices.clear();
  }

  section = vertexpool.section(dim.x*dim.y, 0, glm::vec3(0));

  for(int i = 0; i < dim.x; i++)
  for(int j = 0; j < dim.y; j++)
    update(ivec2(i,j), vertexpool);

  for(int i = 0; i < dim.x-1; i++){
  for(int j = 0; j < dim.y-1; j++){

    vertexpool.indices.push_back(i*dim.y+j);
    vertexpool.indices.push_back(i*dim.y+(j+1));
    vertexpool.indices.push_back((i+1)*dim.y+j);

    vertexpool.indices.push_back((i+1)*dim.y+j);
    vertexpool.indices.push_back(i*dim.y+(j+1));
    vertexpool.indices.push_back((i+1)*dim.y+(j+1));

  }}

  vertexpool.resize(section, vertexpool.indices.size());
  vertexpool.index();
  vertexpool.update();

}

void Layermap::update(ivec2 p, Vertexpool<Vertex>& vertexpool){

  sec* top = dat[p.x*dim.y+p.y];
  while(top != NULL && top->floor > (float)SLICE/(float)SCALE)
    top = top->prev;

  if(top == NULL){
    vertexpool.fill(section, p.x*dim.y+p.y,
      vec3(p.x, 0, p.y),
      vec3(0,1,0),
      soils[soilmap["Air"]].color,
      soilmap["Air"]
    );
  }

  else if(top->floor + top->size > (float)SLICE/(float)SCALE){

    if(top->floor + top->size*top->saturation > (float)SLICE/(float)SCALE)
    vertexpool.fill(section, p.x*dim.y+p.y,
      vec3(p.x, SLICE, p.y),
      vec3(0,1,0),
  //    normal(p),
      mix(soils[soilmap["Air"]].color, soils[top->type].color, 0.6),
      soilmap["Air"]
    );

    else
    vertexpool.fill(section, p.x*dim.y+p.y,
      vec3(p.x, SLICE, p.y),
      vec3(0,1,0),
//    normal(p),
      soils[top->type].color,
      top->type
    );

  }

  else{

/*
    if(top->saturation == 1)  //Fill Watertable!
    vertexpool.fill(section, p.x*dim.y+p.y,
      vec3(p.x, SCALE*(top->floor + top->size), p.y),
      normal(p),
      soils[soilmap["Air"]].color
    );
*/
//    else
    vertexpool.fill(section, p.x*dim.y+p.y,
      vec3(p.x, SCALE*(top->floor + top->size), p.y),
      normal(p),
      soils[top->type].color,
      top->type
    );

  }

  /*

  if(surface(p) == soilmap["Air"])
  vertexpool.fill(section, p.x*dim.y+p.y,
    vec3(p.x, SCALE*height(p), p.y),
    normal(p),
    soils[surface(p)].color
  );
  else
  vertexpool.fill(section, p.x*dim.y+p.y,
    vec3(p.x, SCALE*height(p), p.y),
    normal(p),
    soils[surface(p)].color
  );

  */

}

void Layermap::update(Vertexpool<Vertex>& vertexpool){
  for(int i = 0; i < dim.x; i++)
  for(int j = 0; j < dim.y; j++)
    update(ivec2(i,j), vertexpool);
}

void Layermap::slice(Vertexpool<Vertex>& vertexpool, double s = SCALE){

  for(int i = 0; i < dim.x; i++)
  for(int j = 0; j < dim.y; j++){

    ivec2 p = ivec2(i,j);

    //Find the first element which starts below the scale!
    sec* top = dat[p.x*dim.y+p.y];
    while(top != NULL && top->floor > s/SCALE)
      top = top->prev;

    if(top == NULL){
      vertexpool.fill(section, p.x*dim.y+p.y,
        vec3(p.x, 0, p.y),
        vec3(0,1,0),
        soils[soilmap["Air"]].color,
        soilmap["Air"]
      );
    }

    else if(top->floor + top->size > s/SCALE){
      if(top->floor + top->size*top->saturation > s/SCALE)
      vertexpool.fill(section, p.x*dim.y+p.y,
        vec3(p.x, s, p.y),
        vec3(0,1,0),
        mix(vec4(1,0,0,1), soils[top->type].color, 0.6),
        top->type
      );
      else
      vertexpool.fill(section, p.x*dim.y+p.y,
        vec3(p.x, s, p.y),
        vec3(0,1,0),
        soils[top->type].color,
        top->type
      );
    }

    else{
      if(top->saturation == 0)  //Fill Watertable!
      vertexpool.fill(section, p.x*dim.y+p.y,
        vec3(p.x, SCALE*(top->floor + top->size), p.y),
        normal(p),
        vec4(1,0,0,1),
        top->type
      );
      else
      vertexpool.fill(section, p.x*dim.y+p.y,
        vec3(p.x, SCALE*(top->floor + top->size), p.y),
        normal(p),
        soils[top->type].color,
        top->type
      );
    }

  }
}
