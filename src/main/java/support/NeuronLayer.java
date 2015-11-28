package support;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by gilang on 26/11/2015.
 */
public class NeuronLayer implements Serializable {

    public List<Neuron> neurons;

    public NeuronLayer(){
        neurons = new ArrayList<>();
    }

    public NeuronLayer(List<Neuron> neuronList){
        neurons = neuronList;
    }

    public double getOutput(){
        if(neurons.size() == 1)
            return neurons.get(0).getOutput();
        else{
            int idx = 0;
            double max = 0;
            for(int i=0; i<neurons.size(); i++){
                if(neurons.get(i).getOutput() > max){
                    max = neurons.get(i).getOutput();
                    idx = i;
                }
            }
            return (double)idx;
        }
    }
}
