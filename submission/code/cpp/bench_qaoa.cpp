#include "qaoa_cpu.hpp"
#include <chrono>
#include <complex>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

static std::string arg(int argc,char**argv,const std::string& k,const std::string& d="") { for(int i=1;i+1<argc;++i) if(argv[i]==k) return argv[i+1]; return d; }
int main(int argc,char**argv){
 try{
  int n=std::stoi(arg(argc,argv,"--n")); int p=std::stoi(arg(argc,argv,"--p")); int batch=std::stoi(arg(argc,argv,"--batch","1")); int threads=std::stoi(arg(argc,argv,"--threads","1"));
  auto g=qaoa::load_graph_csv(arg(argc,argv,"--graph"),n); std::vector<std::vector<double>> thetas(batch,std::vector<double>(2*p));
  std::mt19937_64 rng(1234+n+p+batch); std::uniform_real_distribution<double> unif(0.0,3.14159265358979323846);
  for(auto& th: thetas) for(auto& x: th) x=unif(rng);
  qaoa::Timings tim; (void)qaoa::evaluate(g,p,thetas[0],&tim);
  auto t0=std::chrono::steady_clock::now(); auto vals=qaoa::evaluate_batch(g,p,thetas,threads); auto t1=std::chrono::steady_clock::now();
  double ms=std::chrono::duration<double,std::milli>(t1-t0).count(); double msq=ms/batch; double qps=1000.0/msq; std::string out=arg(argc,argv,"--out","bench.csv");
  std::ofstream f(out); f << "timestamp,git_commit,compiler,compiler_version,compiler_flags,os,cpu_model,ram_gb,n,p,dim,batch_size,threads,backend,deterministic_reduction,ms_total,ms_per_query,queries_per_second,speedup_vs_one_thread,parallel_efficiency,memory_bytes,phase_ms,mixer_ms,expectation_ms,reduction_ms,allocation_ms,max_abs_error_vs_python\n";
  f << qaoa::timestamp_utc() << ",unknown,\"" << qaoa::compiler_string() << "\",,\"-O3 -std=c++20 -pthread\",macOS,unknown,unknown," << n << "," << p << "," << 2*p << "," << batch << "," << threads << ",std_thread,true," << ms << "," << msq << "," << qps << ",,," << ((1ull<<n)*sizeof(std::complex<double>)) << "," << tim.phase_ms << "," << tim.mixer_ms << "," << tim.expectation_ms << "," << tim.reduction_ms << "," << tim.allocation_ms << ",\n";
 }catch(const std::exception& e){ std::cerr<<"error: "<<e.what()<<"\n"; return 2; }
}
