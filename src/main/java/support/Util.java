package support;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by gilang on 26/11/2015.
 */
public class Util {

    public static double sigmoid (double outputFromNet){
        return (double)1 / (double)(1 + Math.exp(-outputFromNet));
    }

    public static double errorOutput (double output, double target){
        return (output * (1-output) * (target - output));
    }

    public static double deltaWeightHiddenNeuronToOutput (double learningRate, double errorOutput, double outputHiddenNeuron){
        return learningRate * errorOutput * outputHiddenNeuron;
    }

    public static double errorHiddenNeuron (double outputHiddenNeuron, double weightHiddenNeuron, double errorOutput){
        return (outputHiddenNeuron * (1-outputHiddenNeuron) * (weightHiddenNeuron*errorOutput));
    }

    public static double deltaWeightInputToHidden (double learningRate, double errorHiddenNeuron, double input){
        return learningRate * errorHiddenNeuron * input;
    }

    public static double sign (double outputFromNet){
        if (outputFromNet < 0) {
            return -1;
        } else {
            return 1;
        }
    }

    public static double fungsiNet(List<Neuron> inputs, List<Double> weights) throws Exception{
        if (inputs.size() != weights.size()) {
            throw new Exception(inputs.size() + " " + weights.size());
        }

        double output = 0;

        for (int i=0 ; i < weights.size() ; i++) {
            output += inputs.get(i).getOutput() * weights.get(i);
        }

        return output;
    }

    public static double deltaWeight (double learningRate, double target, double output, double input){
        return (learningRate * (target - output) * input);
    }

    public static double MSE (List<Double> targets, List<Double> outputs) throws Exception{
        if (targets.size() != outputs.size()) {
            throw new Exception(targets.size() + " " + outputs.size());
        }

        double MSE = 0;
        for (int i=0 ; i < targets.size() ; i++) {
            MSE += Math.pow(targets.get(i) - outputs.get(i), 2);
        }
        return MSE/(double)2;
    }

    public static double multiDeltaWeight (double learningRate, List<Double> targets, List<Double> outputs, List<Double> inputs) throws Exception{
        if (targets.size() != outputs.size() || targets.size() != inputs.size()) {
            throw new Exception();
        }

        double sumDeltaWeight = 0;
        for (int i=0 ; i < targets.size() ; i++) {
            sumDeltaWeight += deltaWeight (learningRate, targets.get(i), outputs.get(i), inputs.get(i));
        }
        return sumDeltaWeight;
    }
}
