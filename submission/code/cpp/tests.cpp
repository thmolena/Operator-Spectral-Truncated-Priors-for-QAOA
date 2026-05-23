#include "qaoa_cpu.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
int main(){
    qaoa::Graph g; g.n=3; g.edges={{0,1},{1,2}};
    for(int p: {3,6,7}){
        std::vector<double> th(2*p,0.2);
        double a=qaoa::evaluate(g,p,th,nullptr); double b=qaoa::evaluate(g,p,th,nullptr);
        assert(std::isfinite(a)); assert(std::abs(a-b)<1e-12);
        auto vals=qaoa::evaluate_batch(g,p,std::vector<std::vector<double>>{th,th,th,th},2);
        for(double v: vals) assert(std::abs(v-a)<1e-12);
    }
    bool failed=false; try{ qaoa::evaluate(g,3,std::vector<double>(5),nullptr); }catch(const std::invalid_argument&){ failed=true; }
    assert(failed); std::cout<<"C++ QAOA CPU tests passed\n";
}
