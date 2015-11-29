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
    private double errorOutput;
//    private double deltaWeight;
    private double output;
    private double error;
    private Random r;

    public Neuron(){
        inputsNeuron = new ArrayList<>();
        inputsWeight = new ArrayList<>();
        error = 0;
        output = 0;
        errorOutput = 0;
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

    public List<Neuron> getInputsNeuron() { return inputsNeuron; }

    public void initWeight(int inputNum){
        inputsWeight = new ArrayList<>();
        for(int i=0; i<inputNum; i++) {
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
//        System.out.println();
        if (functionType == MyANN.FUNCTION_SIGMOID) {
            output = Util.sigmoid(net);
        }else if(functionType == MyANN.FUNCTION_SIGN){
            output = Util.sign(net);
        } else if(functionType == MyANN.FUNCTION_TYPE_3) {
            output = net;
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

    /* Update input's weight. Used for any neurons except for output neuron */
    public void updateWeight(){
        for(int i=0; i<inputsNeuron.size(); i++){
            Neuron n = inputsNeuron.get(i);
            double newWeight = inputsWeight.get(i) + (error * learningRate * n.getOutput());
            inputsWeight.set(i, newWeight);
            n.setError((n.getOutput() * (1-n.getOutput()) * inputsWeight.get(i) * error) + n.getError());
//            System.out.println(newWeight + " " + error);
        }
    }

    /* Update input's weight. Used only for output neuron */
    public void updateOutputWeight(){
        for(int i=0; i<inputsNeuron.size(); i++){
            Neuron n = inputsNeuron.get(i);
            double newWeight = inputsWeight.get(i) + (error * learningRate * n.getOutput());
            inputsWeight.set(i, newWeight);
            n.setError(n.getOutput() * (1-n.getOutput()) * inputsWeight.get(i) * error);
        }
    }


    public List<Double> getDeltaWeights (double targetOutput) {
        ArrayList<Double> deltaWeights = new ArrayList<>();
        for(int i=0; i<inputsNeuron.size(); i++){
            Neuron inputNeuron = inputsNeuron.get(i);
            double deltaWeight =  Util.deltaWeight(learningRate, targetOutput, output, inputNeuron.getOutput());
            deltaWeights.add(i, deltaWeight);
        }
        return deltaWeights;
    }

    public void updateWeightSingleLayer (double targetOutput) {
//        System.out.println("inputneuron: " + inputsNeuron.size());
        for(int i=0; i<inputsNeuron.size(); i++){
            Neuron inputNeuron = inputsNeuron.get(i);
            double newWeight = inputsWeight.get(i) + Util.deltaWeight(learningRate, targetOutput, output, inputNeuron.getOutput());
//            System.out.println("learningRate: " + learningRate + " targetoutput: " + targetOutput + " output: " + output + " inputNeuron : " + inputNeuron.getOutput());
//            System.out.println("deltaWeight " + Util.deltaWeight(learningRate, targetOutput, output, inputNeuron.getOutput()));
            inputsWeight.set(i, newWeight);
        }
    }

    public void updateWeightCumulative(List<Double> deltaWeights) {
        for (int i=0; i<inputsNeuron.size(); i++) {
            double newWeight = inputsWeight.get(i) + deltaWeights.get(i);
            inputsWeight.set(i, newWeight);
        }
    }

    /* Reset this neuron output value, error, and nextNeuron output for computing next instance */
    public void reset(){
        output = 0;
        error = 0;
        errorOutput = 0;
    }

    /* Reset this neuron inputs. Used only for first layer neuron */
    public void resetInput(){
        Neuron bias  = inputsNeuron.get(0);
        inputsNeuron = new ArrayList<>();
        inputsNeuron.add(bias);
    }

    public List<Double> getWeights(){
        return inputsWeight;
    }
}
