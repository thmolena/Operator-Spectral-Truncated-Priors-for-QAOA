#include "qaoa_cpu.hpp"
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

static std::string arg(int argc,char**argv,const std::string& k,const std::string& d="") { for(int i=1;i+1<argc;++i) if(argv[i]==k) return argv[i+1]; return d; }
static std::vector<double> parse_theta(const std::string& s){ std::vector<double> v; std::stringstream ss(s); std::string x; while(std::getline(ss,x,',')) v.push_back(std::stod(x)); return v; }
int main(int argc,char**argv){
    try{
        int n=std::stoi(arg(argc,argv,"--n")); int p=std::stoi(arg(argc,argv,"--p"));
        auto g=qaoa::load_graph_csv(arg(argc,argv,"--graph"),n); auto th=parse_theta(arg(argc,argv,"--theta"));
        std::cout.setf(std::ios::fixed); std::cout.precision(12); std::cout << qaoa::evaluate(g,p,th,nullptr) << "\n";
    }catch(const std::exception& e){ std::cerr<<"error: "<<e.what()<<"\n"; return 2; }
}
