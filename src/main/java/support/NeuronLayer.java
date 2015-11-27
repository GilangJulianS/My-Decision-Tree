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
        return neurons.get(0).getOutput();
    }
}
