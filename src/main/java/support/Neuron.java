package support;

import classifier.MyANN;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Created by gilang on 26/11/2015.
 */
public class Neuron implements Serializable{


    private List<Neuron> inputsNeuron;
    private List<Double> inputsWeight;
    private static double initialWeight;
    private static int mode;
    private static int functionType;
    private static double learningRate;
    private static double momentum;
    private double output;
    private double error;
    private Random r;

    public Neuron(){
        inputsNeuron = new ArrayList<>();
        inputsWeight = new ArrayList<>();
        error = 0;
        output = 0;
        r = new Random();
    }

    public static void setInitialWeight(double weight){
        initialWeight = weight;
    }

    public static void setMode(int classifierMode){
        mode = classifierMode;
    }

    public static void setLearningRate(double rate){
        learningRate = rate;
    }

    public static void setMomentum(double momen){
        momentum = momen;
    }

    public static void setActivationFunction(int activationFunctionType){
        functionType = activationFunctionType;
    }

    public void addInputs(List<Neuron> inputs){
        inputsNeuron.addAll(inputs);
    }

    public void initWeight(){
        inputsWeight = new ArrayList<>();
        for(int i=0; i<inputsNeuron.size(); i++) {
            if (initialWeight == MyANN.RANDOM_WEIGHT) {
                inputsWeight.add(r.nextDouble());
            } else {
                inputsWeight.add(initialWeight);
            }
        }
    }

    public void addInput(Neuron input){
        inputsNeuron.add(input);
    }

    public void computeOutput() throws Exception {
        double net = Util.fungsiNet(inputsNeuron, inputsWeight);
        for(int i=0; i<inputsNeuron.size(); i++){
//            System.out.print("input " + inputsNeuron.get(i).getOutput() + " ");
        }
        System.out.println();
        if (functionType == MyANN.FUNCTION_SIGMOID) {
            output = Util.sigmoid(net);
        }else if(functionType == MyANN.FUNCTION_SIGN){
            output = Util.sign(net);
        }
    }

    public Neuron setOutput(double output){
        this.output = output;
        return this;
    }

    public double getOutput(){
        return output;
    }

    public void setError(double error){
        this.error = error;
    }

    public double getError(){
        return error;
    }

    public void updateWeight(boolean isLastLayer){
        for(int i=0; i<inputsNeuron.size(); i++){
            Neuron n = inputsNeuron.get(i);
            if(isLastLayer) {
                n.setError(inputsWeight.get(i) * error);
            }else{
                n.setError(inputsWeight.get(i) * error + n.getError());
            }
            double newWeight = inputsWeight.get(i) + (error * learningRate * n.getOutput());
            inputsWeight.set(i, newWeight);
            System.out.println(newWeight + " " + error);
        }
    }

    public void updateOutputWeight(){
        for(int i=0; i<inputsNeuron.size(); i++){
            Neuron n = inputsNeuron.get(i);
            double newWeight = inputsWeight.get(i) + (error * learningRate * n.getOutput());
            inputsWeight.set(i, newWeight);
        }
    }

    public void reset(){
        output = 0;
        error = 0;
    }

    public void resetInput(){
        inputsNeuron = new ArrayList<>();
    }

    public List<Double> getWeights(){
        return inputsWeight;
    }

}