package support;

import classifier.*;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.trees.Id3;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.supervised.instance.Resample;
import weka.filters.unsupervised.attribute.Remove;

import java.util.Random;

/**
 * Created by gilang on 28/09/2015.
 */
public class WekaIFace {

    public static final int BAYES = 0;
    public static final int ID3 = 1;
    public static final int J48 = 2;
    public static final int MY_BAYES = 3;
    public static final int MY_ID3 = 4;
    public static final int MY_J48 = 5;
    public static final int MY_ANN = 6;
    public static final int PTR = 7;
    public static final int BATCH = 8;
    public static final int DELTA_RULE = 9;

    public static Instances readArff(String fileName) throws Exception {
        // read file
        DataSource source = new DataSource(fileName);
        Instances data = source.getDataSet();
        // set class index if not set
        if(data.classIndex() == -1){
            data.setClassIndex(data.numAttributes() - 1);
        }
        return data;
    }

    public static Instances removeAttribute(Instances data, String attribute) throws Exception {
        Remove remove = new Remove();
        remove.setAttributeIndices(attribute);
        remove.setInputFormat(data);
        Instances filterData = Filter.useFilter(data, remove);
        return  filterData;
    }

    public static Instances resampleData(Instances data) throws Exception {
        Resample resample = new Resample();
        resample.setInputFormat(data);
        Instances filterData = Filter.useFilter(data, resample);
        return filterData;
    }

    public static Classifier buildClassifier(Instances data, int classifierType, boolean prune) throws Exception {
        Classifier classifier = null;
        if(classifierType == BAYES){
            classifier = new NaiveBayes();
            classifier.buildClassifier(data);
        }else if(classifierType == ID3){
            classifier = new Id3();
            classifier.buildClassifier(data);
        }else if(classifierType == J48){
            classifier = new J48();
            classifier.buildClassifier(data);
        }else if(classifierType == MY_ID3){
            classifier = new MyID3();
            classifier.buildClassifier(data);
        }else if(classifierType == MY_J48){
            classifier = new MyJ48();
            ((MyJ48)classifier).enablePrune(prune);
            classifier.buildClassifier(data);
        }else if(classifierType == MY_ANN){
//            classifier = new MyANN(MyANN.MODE_MULTILAYER_PERCEPTRON, MyANN.FUNCTION_SIGMOID, 0d, "2", 0.1d, 0d, 10000, 0.01d);
            classifier = new MyANN(MyANN.MODE_BATCH_GRADIENT_DESCENT, 0d, "", 0.2d, 0.1d, 5000, 0.01d);
//            classifier = new PerceptronTrainingRule(0d, 0.1d, 0d, -1, 0.01d);
//            classifier = new DeltaRuleBatch(0d, 0.1d, 0d, -1, 0.01d);
//            classifier = new DeltaRuleIncremental(0d, 0.1d, 0d, -1, 0.01d);
            classifier.buildClassifier(data);
//            for(int i=0; i<data.numInstances(); i++){
//                classifier.classifyInstance(data.instance(i));
//            }
        }
        return classifier;
    }

    public static void crossValidate(Instances data, Classifier classifier) throws Exception {
        // evaluate set
        Evaluation evaluation = new Evaluation(data);
        evaluation.crossValidateModel(classifier, data, 10, new Random(1));
        System.out.println(evaluation.toSummaryString("\n----- Cross Validation Summary -----\n\n", false));
    }

    public static void evaluateTestSet(Instances data, Classifier classifier, Instances testSet) throws Exception {
        // evaluate set
        Evaluation evaluation = new Evaluation(data);
        evaluation.evaluateModel(classifier, testSet);
        System.out.println(evaluation.toSummaryString("\n----- TestSet Evaluation Summary -----\n\n", false));
    }

    public static void evaluateSplit(Instances data, Classifier classifier, float trainPercentage) throws Exception {
        Instances newData = new Instances(data);
        newData.randomize(new Random(1));

        // split data to training set and test set
        int trainSize = Math.round(data.numInstances() * (trainPercentage/100));
        int testSize = data.numInstances() - trainSize;
        Instances dataTrain = new Instances(data, 0, trainSize);
        Instances dataTest = new Instances(data, trainSize, testSize);

        // evaluate set
        classifier.buildClassifier(dataTrain);
        evaluateTestSet(dataTrain, classifier, dataTest);
    }

    public static Classifier loadModel(String fileName) throws Exception {
        return (Classifier) SerializationHelper.read(fileName);
    }

    public static void saveModel(String fileName, Classifier classifier) throws Exception {
        SerializationHelper.write(fileName, classifier);
    }

    public static Instances classifyInstances(Classifier classifier, Instances data) throws Exception {
        Instances labeledData = new Instances(data);
        // labeling data
        for (int i = 0; i < labeledData.numInstances(); i++) {
            double clsLabel = classifier.classifyInstance(data.instance(i));
            labeledData.instance(i).setClassValue(clsLabel);
        }
        return labeledData;
    }
}
