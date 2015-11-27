package classifier;

import support.Neuron;
import support.NeuronLayer;
import support.Util;
import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.Instance;
import weka.core.Instances;

import javax.tools.Tool;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by gilang on 25/11/2015.
 */
public class MyANN extends Classifier {

    public static final int FUNCTION_SIGMOID = 100;
    public static final int FUNCTION_SIGN = 101;
    public static final int ACTIVATION_FUNCTION_3 = 102;
    public static final int PERCEPTRON_TRAINING_RULE = 100;
    public static final int DELTA_RULE = 101;
    public static final int BATCH_GRADIENT_DESCENT = 102;
    public static final int MULTILAYER_PERCEPTRON = 103;
    public static final double RANDOM_WEIGHT = -999;
    private List<NeuronLayer> layers;
    private List<Double> targetOutputs;
    private List<Double> outputs;
    private int[] neuronsNumber;
    private double initialWeight;
    private double mse;
    private double mseThreshold;
    private double learningRate;
    private double momentum;
    private int iteration;
    private int maxIteration;
    private int mode;
    private int functionType;

    public MyANN(int mode, int functionType, double initialWeight, String neuronsCount, double learningRate, double momentum, int maxIteration, double mseThreshold){
        this.mode = mode;
        this.functionType = functionType;
        this.initialWeight = initialWeight;
        this.mseThreshold = mseThreshold;
        this.maxIteration = maxIteration;
        this.learningRate = learningRate;
        this.momentum = momentum;
        targetOutputs = new ArrayList<>();
        mse = Double.POSITIVE_INFINITY;
        String[] temp = neuronsCount.split(",");
        neuronsNumber = new int[temp.length];
        int i=0;
        for(String s : temp){
            neuronsNumber[i] = Integer.valueOf(s);
            i++;
        }
    }

    @Override
    public Capabilities getCapabilities() {
        Capabilities capabilities = super.getCapabilities();
        capabilities.disableAll();

        capabilities.enable(Capability.BINARY_CLASS);
        capabilities.enable(Capability.NOMINAL_CLASS);
        capabilities.enable(Capability.NUMERIC_CLASS);
        capabilities.enable(Capability.MISSING_CLASS_VALUES);
        capabilities.enable(Capability.DATE_CLASS);
        capabilities.enable(Capability.BINARY_ATTRIBUTES);
        capabilities.enable(Capability.DATE_ATTRIBUTES);
        capabilities.enable(Capability.UNARY_ATTRIBUTES);
        capabilities.enable(Capability.EMPTY_NOMINAL_ATTRIBUTES);
        capabilities.enable(Capability.NUMERIC_ATTRIBUTES);
        capabilities.enable(Capability.MISSING_VALUES);
        capabilities.enable(Capability.NOMINAL_ATTRIBUTES);

        return capabilities;
    }

    @Override
    public void buildClassifier(Instances instances) throws Exception {
        initStructure();
        printPerceptron();
        iteration = 0;
        for(int i=0; i<instances.numInstances(); i++){
            targetOutputs.add(instances.instance(i).classValue());
//            System.out.println(instances.instance(i).classValue());
        }
        while(true){
            System.out.println(mse + " " + iteration);
            if(mseThreshold == -1 && maxIteration == -1)
                break;
            if(mseThreshold != -1)
                if(mse <= mseThreshold)
                    break;
            if(maxIteration != -1)
                if(iteration >= maxIteration)
                    break;

            outputs = new ArrayList<>();
            //satu epoch
            for(int i=0; i<instances.numInstances(); i++){
                computeForward(instances.instance(i));

//                System.out.println("output " + layers.get(layers.size()-1).getOutput());

                backProp(instances.instance(i));
                outputs.add(layers.get(layers.size()-1).getOutput());
                //reset neuron
                for(NeuronLayer nl : layers){
                    for(Neuron n : nl.neurons){
                        n.reset();
                    }
                }
                for(Neuron n : layers.get(0).neurons){
                    n.resetInput();
                }
            }
            mse = Util.MSE(targetOutputs, outputs);
            iteration++;
        }
    }

    public void computeForward(Instance instance) throws Exception {
        //add input to each first layer node
        List<Neuron> inputs = new ArrayList<>();
        for(int i=0; i<instance.numAttributes()-1; i++){
            inputs.add(new Neuron().setOutput(instance.value(i)));
        }
        for(Neuron n : layers.get(0).neurons){
            n.addInputs(inputs);
            n.initWeight();
        }
        for(NeuronLayer layer : layers){
            for (Neuron n : layer.neurons) {
                n.computeOutput();
            }
        }
        for(NeuronLayer layer : layers){
            for (Neuron n : layer.neurons) {
//                System.out.print(n.getOutput() + " ");
            }
//            System.out.println();
        }
    }

    public void backProp(Instance instance){
        Neuron outputNeuron = layers.get(layers.size()-1).neurons.get(0);
        double target = instance.classValue();
        double output = outputNeuron.getOutput();
        double errorOutput = Util.errorOutput(output, target);

        System.out.println(output + " " + target + " " + errorOutput);
        outputNeuron.setError(errorOutput);
        outputNeuron.updateOutputWeight();
//        for(int j=0; j<outputNeuron.getWeights().size(); j++){
//            System.out.println("weight " + outputNeuron.getWeights().get(j));
//        }

        for(int i=layers.size()-2; i>=0; i--){
            for(Neuron n : layers.get(i).neurons){
                n.updateWeight();
//                for(int j=0; j<n.getWeights().size(); j++){
//                    System.out.println("weight " + n.getWeights().get(j));
//                }
            }
        }
    }

    public void initStructure(){
        Neuron.setInitialWeight(initialWeight);
        Neuron.setMode(mode);
        Neuron.setActivationFunction(functionType);
        Neuron.setLearningRate(learningRate);
        Neuron.setMomentum(momentum);
        layers = new ArrayList<>();
        //iterasi layer, layer terakhir adalah output
        for(int i=0; i<=neuronsNumber.length; i++){
            List<Neuron> neurons = new ArrayList<>();

            //add neuron ke layer
            if(i != neuronsNumber.length) {
                for (int j = 0; j < neuronsNumber[i]; j++) {
                    neurons.add(new Neuron());
                }
            }else{
                neurons.add(new Neuron());
            }
            layers.add(new NeuronLayer(neurons));

            if(i>0) {
                for (Neuron n : neurons) {
                    n.addInputs(layers.get(i - 1).neurons);
                    n.initWeight();
                }
            }
        }
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        return super.classifyInstance(instance);
    }

    public void printPerceptron(){
        for(NeuronLayer layer : layers){
            for(Neuron n : layer.neurons){
                System.out.print("() ");
            }
            System.out.println();
        }
    }
}
