package classifier;

import support.Neuron;
import support.NeuronLayer;
import support.Util;
import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SystemInfo;

import javax.tools.Tool;
import java.util.*;

/**
 * Created by gilang on 25/11/2015.
 */
public class MyANN extends Classifier {

    public static final int FUNCTION_SIGMOID = 100;
    public static final int FUNCTION_SIGN = 101;
    public static final int FUNCTION_NET = 102;
    public static final int MODE_PERCEPTRON_TRAINING_RULE = 100;
    public static final int MODE_DELTA_RULE_INCREMENTAL = 101;
    public static final int MODE_BATCH_GRADIENT_DESCENT = 102;
    public static final int MODE_MULTILAYER_PERCEPTRON = 103;
    public static final double RANDOM_WEIGHT = -999;
    private List<NeuronLayer> layers;
    private List<Double> targetOutputs;
    private List<Double> outputs;
    private List<Double> deltaWeights;
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
    private int numInstance;
	private double minMSE = Double.MAX_VALUE;

    public MyANN(int mode, double initialWeight, String neuronsCount, double learningRate, double momentum, int maxIteration, double mseThreshold){
        this.mode = mode;
	    if(mode == MODE_MULTILAYER_PERCEPTRON)
            functionType = FUNCTION_SIGMOID;
	    else if(mode == MODE_PERCEPTRON_TRAINING_RULE)
		    functionType = FUNCTION_SIGN;
	    else
	        functionType = FUNCTION_NET;
        this.initialWeight = initialWeight;
        this.mseThreshold = mseThreshold;
        this.maxIteration = maxIteration;
        this.learningRate = learningRate;
        this.momentum = momentum;
        if(mode == MODE_MULTILAYER_PERCEPTRON) {
	        String[] temp = neuronsCount.split(",");
	        neuronsNumber = new int[temp.length];
	        int i = 0;
	        for (String s : temp) {
		        neuronsNumber[i] = Integer.valueOf(s);
		        i++;
	        }
        }else{
	        neuronsNumber = new int[0];
        }
    }

    @Override
    public Capabilities getCapabilities() {
        Capabilities capabilities = super.getCapabilities();
        capabilities.disableAll();

        capabilities.enable(Capability.BINARY_CLASS);
        capabilities.enable(Capability.NOMINAL_CLASS);
        capabilities.enable(Capability.MISSING_CLASS_VALUES);
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
    public void buildClassifier(Instances instances) throws Exception {;
        mse = Double.POSITIVE_INFINITY;
	    targetOutputs = new ArrayList<>();
        numInstance = instances.numInstances();
        initStructure(instances);
        printPerceptron();
        iteration = 0;
        deltaWeights = new ArrayList<>();
        double deltaWeight;
        List<Double> temp = new ArrayList<>();
        for(int i=0; i<instances.numInstances(); i++){
            targetOutputs.add(instances.instance(i).classValue());
//            System.out.println(instances.instance(i).classValue());
        }
        while(true){
//            if(iteration % 1000 == 0)
                System.out.println(mse + " " + iteration);
//	        if(mse < minMSE)
//		        minMSE = mse;
//	        System.out.println("min MSE " + minMSE);
	        if(mseThreshold == -1 && maxIteration == -1)
                break;
            if(mseThreshold != -1)
                if(mse <= mseThreshold)
                    break;
            if(maxIteration != -1)
                if(iteration >= maxIteration)
                    break;
	        //satu epoch
            for(int i=0; i<instances.numInstances(); i++){
                computeForward(instances.instance(i));

//                if(iteration % 1000 == 0)
//                    System.out.println("\r\n>>>>>output - target " + layers.get(layers.size()-1).getOutput() + " " + targetOutputs.get(i));

                backProp(instances.instance(i), i);
                //print weight
//                if(iteration % 1000000 == 0) {
//                    for (int j = 0; j < layers.size(); j++) {
//                        int ni = 0;
//                        for (Neuron n : layers.get(j).neurons) {
//                            System.out.println("Neuron layer " + j + " ke-" + ni + " ==> error : " + n.getError());
//                            System.out.println("output : " + n.getOutput());
//                            for (double d : n.getWeights()) {
//                                System.out.print("weight " + d + ", ");
//                            }
//                            System.out.println();
//                            ni++;
//                        }
//                    }
//                }
                if (mode == MODE_BATCH_GRADIENT_DESCENT) {
                    for(Neuron n : layers.get(0).neurons) {
                        if (i == 0) {
                            for (int k = 0; k < n.getInputsNeuron().size(); k++) {
                                temp.add(k, 0d);
                                //                        System.out.println("temp" + temp.get(k));
                            }
                        }
                        for (int j = 0; j < n.getInputsNeuron().size(); j++) {
                            deltaWeight = n.getDeltaWeights(targetOutputs.get(i)).get(j);
                            deltaWeights.add(j, temp.get(j) + deltaWeight);
                            temp.set(j, deltaWeights.get(j));
                        }

                        if (i == instances.numInstances() - 1) {
                            n.updateWeightCumulative(deltaWeights, i);
                        }
                    }
                }
                //reset neuron
                resetNeurons();
            }
            //recomputing outputs
            outputs = new ArrayList<>();
            for(int i=0; i<instances.numInstances(); i++){
                computeForward(instances.instance(i));
                outputs.add(layers.get(layers.size()-1).getOutput());
//                if(iteration % 1000 == 0)
//                    System.out.println("output - target " + outputs.get(i) + " " + targetOutputs.get(i));
                resetNeurons();
            }
            mse = Util.MSE(targetOutputs, outputs);
            iteration++;
        }
    }

    public void resetNeurons(){
	    for(NeuronLayer nl : layers){
            for(Neuron n : nl.neurons){
                n.reset();
            }
        }
        for(Neuron n : layers.get(0).neurons){
            n.resetInput();
        }
    }

    public void computeForward(Instance instance) throws Exception {
        //add input to each first layer node
        List<Neuron> inputs = new ArrayList<>();

        //adding inputs
	    for(int i=0; i<instance.numAttributes()-1; i++){
            inputs.add(new Neuron(numInstance).setOutput(instance.value(i)));
        }
        for(Neuron n : layers.get(0).neurons){
	        n.addInputs(inputs);
        }

        //computing output of each node
	    for(NeuronLayer layer : layers){
		    for (Neuron n : layer.neurons) {
                n.computeOutput();
            }
        }
    }

    public void backProp(Instance instance, int instanceNumber){
        //System.out.println("tes");
        double target = instance.classValue();

        //iterate output neurons
        for(int i=0; i<layers.get(layers.size()-1).neurons.size(); i++){
            Neuron n = layers.get(layers.size()-1).neurons.get(i);

            // if neuron index == target, then error = 0
            // else error = 1
	        if((double)i == target){
                n.setError(Util.errorOutput(n.getOutput(), 1d, mode));
	        }else{
		        if(functionType == FUNCTION_SIGN) {
			        n.setError(Util.errorOutput(n.getOutput(), -1d, mode));
		        }else{
			        n.setError(Util.errorOutput(n.getOutput(), 0, mode));
		        }
            }
//	        System.out.println("output " + n.getOutput() + " error " + n.getError());
            if(mode == MODE_BATCH_GRADIENT_DESCENT) {
                // do nothing
            } else {
                n.updateOutputWeight(instanceNumber, mode);
            }
        }

        for(int i=layers.size()-2; i>=0; i--){
            for(Neuron n : layers.get(i).neurons){
                n.updateWeight(instanceNumber);
//                for(int j=0; j<n.getWeights().size(); j++){
//                    System.out.println("weight " + n.getWeights().get(j));
//                }
            }
        }
    }

    public void initStructure(Instances instances){
        int numAttribute = instances.numAttributes()-1;
        int numOutput = instances.classAttribute().numValues();
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
                    neurons.add(new Neuron(numInstance));
                }
            }else{ //add output neuron
                for(int j = 0; j<numOutput; j++) {
                    neurons.add(new Neuron(numInstance));
                }
            }
            layers.add(new NeuronLayer(neurons));
            for (Neuron n : neurons) {
                if(i>0) {
                    n.addInputs(layers.get(i - 1).neurons);
                    n.initWeight(layers.get(i - 1).neurons.size()+1);
                    System.out.println(layers.get(i - 1).neurons.size());
                }else{
                    n.initWeight(numAttribute + 1);
                }
            }
        }
        for(NeuronLayer layer : layers){
            for(Neuron n : layer.neurons){
                Neuron bias = new Neuron(numInstance).setOutput(1);
                n.addInput(bias);
            }
        }
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        resetNeurons();
        computeForward(instance);
//        System.out.println("classifying "  + layers.get(layers.size()-1).getOutput());
        return layers.get(layers.size()-1).getOutput();
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
