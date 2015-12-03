package classifier;

import support.Neuron;
import support.NeuronLayer;
import support.Util;
import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.Instance;
import weka.core.Instances;

import java.util.List;
import java.util.ArrayList;

/**
 * Created by Windy Amelia on 11/28/2015.
 */
public class DeltaRuleIncremental extends Classifier {
    public static final int MODE_DELTA_RULE = 101;
    public static final int FUNCTION_TYPE_3 = 102;
    Neuron outputNeuron;
    private List<Double> targetOutputs;
    private List<Double> outputs;
    private double initialWeight;
    private double mse;
    private double learningRate;
    private double momentum;
    private int iteration;
    private int maxIteration;
    private double mseThreshold;
    private int numInstance;

    public DeltaRuleIncremental(double initialWeight, double learningRate, double momentum, int maxIteration, double mseThreshold) {
        this.initialWeight = initialWeight;
        this.learningRate = learningRate;
        this.momentum = momentum;
        this.maxIteration = maxIteration;
        this.mseThreshold = mseThreshold;
        mse = Double.POSITIVE_INFINITY;
    }

    @Override
    public Capabilities getCapabilities() {
        Capabilities capabilities = super.getCapabilities();
        capabilities.disableAll();

        capabilities.enable(Capabilities.Capability.BINARY_CLASS);
        capabilities.enable(Capabilities.Capability.NOMINAL_CLASS);
        capabilities.enable(Capabilities.Capability.NUMERIC_CLASS);
        capabilities.enable(Capabilities.Capability.MISSING_CLASS_VALUES);
        capabilities.enable(Capabilities.Capability.DATE_CLASS);
        capabilities.enable(Capabilities.Capability.BINARY_ATTRIBUTES);
        capabilities.enable(Capabilities.Capability.DATE_ATTRIBUTES);
        capabilities.enable(Capabilities.Capability.UNARY_ATTRIBUTES);
        capabilities.enable(Capabilities.Capability.EMPTY_NOMINAL_ATTRIBUTES);
        capabilities.enable(Capabilities.Capability.NUMERIC_ATTRIBUTES);
        capabilities.enable(Capabilities.Capability.MISSING_VALUES);
        capabilities.enable(Capabilities.Capability.NOMINAL_ATTRIBUTES);

        return capabilities;
    }

    @Override
    public void buildClassifier(Instances instances) throws Exception {
        targetOutputs = new ArrayList<>();
        numInstance = instances.numInstances();
        outputNeuron = new Neuron(numInstance);
        initStructure(instances);
        iteration=0;

        for(int i=0; i<instances.numInstances(); i++) {
            targetOutputs.add(instances.instance(i).classValue());
//            System.out.println(" toutput: " + targetOutputs.get(i));
        }

        while(true) {
//            if(iteration % 1000000 == 0)
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
//            List<Double> deltaWeights = new ArrayList<>();
//            int numNeuron;
            for(int i=0; i<instances.numInstances(); i++) {
                computeForward(instances.instance(i));
//                outputs.add(outputNeuron.getOutput());
                outputNeuron.updateWeightSingleLayer(targetOutputs.get(i),i);
                outputNeuron.resetInput();
                outputNeuron.reset();
            }

            //recomputing outputs
            outputs = new ArrayList<>();
            for(int i=0; i<instances.numInstances(); i++){
                computeForward(instances.instance(i));
                outputs.add(outputNeuron.getOutput());
                outputNeuron.resetInput();
                outputNeuron.reset();
            }
            mse = Util.MSE(targetOutputs, outputs);
            iteration++;
        }
    }

    public void computeForward(Instance instance) throws Exception {
        List<Neuron> inputs = new ArrayList<>();
        for(int i=0; i<instance.numAttributes()-1; i++){
            inputs.add(new Neuron(numInstance).setOutput(instance.value(i)));
        }
        outputNeuron.addInputs(inputs);
        outputNeuron.computeOutput();
    }

    public double classifyInstance(Instance instance) throws Exception {
        outputNeuron.resetInput();
        outputNeuron.reset();
        computeForward(instance);
//        System.out.println("weight " + )
        System.out.println(">>" + outputNeuron.getOutput());
        return outputNeuron.getOutput();
    }

    public void initStructure(Instances instances) {
        int numAttribute = instances.numAttributes()-1;
        Neuron.setInitialWeight(initialWeight);
        Neuron.setMode(MODE_DELTA_RULE);
        Neuron.setActivationFunction(FUNCTION_TYPE_3);
        Neuron.setLearningRate(learningRate);
        Neuron.setMomentum(momentum);

        outputNeuron.initWeight(numAttribute + 1);
        Neuron bias = new Neuron(numInstance).setOutput(1);
        outputNeuron.addInput(bias);
    }
}
