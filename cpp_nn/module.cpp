#include "module.h"

void Module::backward(const Loss &loss)
{
    float_batch delta = loss.delta;
//    cout<<delta[0][0]<<endl;
    size_t num_layers = this->parameters.size();
    for(int i = num_layers - 1; i >= 0; i--){
        delta = this->parameters[i]->backward(delta);
    }

}
