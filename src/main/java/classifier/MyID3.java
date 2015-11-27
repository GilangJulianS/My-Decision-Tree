package classifier;

import support.WekaIFace;
import weka.classifiers.Classifier;
import weka.core.*;
import weka.core.Capabilities.Capability;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Add;

import java.util.*;

import static weka.core.Utils.log2;

/**
 * Created by gilang on 28/09/2015.
 */
public class MyID3 extends Classifier {

    private MyID3[] successors;
    private Attribute splitAttribute;
    private double classValue;
    private double[] classDistribution;
    private Attribute classAttribute;

    @Override
    public Capabilities getCapabilities() {
        Capabilities capabilities = super.getCapabilities();
        capabilities.disableAll();

        capabilities.enable(Capability.NOMINAL_ATTRIBUTES);
        capabilities.enable(Capability.NUMERIC_ATTRIBUTES);
        capabilities.enable(Capability.BINARY_CLASS);
        capabilities.enable(Capability.NOMINAL_CLASS);
        capabilities.enable(Capability.MISSING_CLASS_VALUES);

        capabilities.setMinimumNumberInstances(0);
        return capabilities;
    }

    @Override
    public void buildClassifier(Instances instances) throws Exception {
        // Mengecek dapatkah classifier menghandle data
        getCapabilities().testWithFail(instances);

        // menghapus instance dengan missing class
        instances.deleteWithMissingClass();
        Instances nominalInstances = toNominal(instances);
        makeTree(nominalInstances);
    }

    public void makeTree(Instances instances) throws Exception {
        // Mengecek ada tidaknya instance yang mencapai node ini
        if (instances.numInstances() == 0) {
            splitAttribute = null;
            classValue = Instance.missingValue();
            classDistribution = new double[instances.numClasses()];
            return;
        } else {
            // Mencari IG maksimum dari atribut
            double[] infoGains = new double[instances.numAttributes()];
            Enumeration attEnum = instances.enumerateAttributes();
            while (attEnum.hasMoreElements()) {
                Attribute att = (Attribute) attEnum.nextElement();
                infoGains[att.index()] = computeIG(instances, att);
            }
            splitAttribute = instances.attribute(indexWithMaxValue(infoGains));

            // Jika IG max = 0, buat daun dengan label kelas mayoritas
            // Jika tidak, buat successor
            if (equalValue(infoGains[splitAttribute.index()], 0)) {
                splitAttribute = null;
                classDistribution = new double[instances.numClasses()];

                for (int i = 0; i < instances.numInstances(); i++) {
                    Instance inst = (Instance) instances.instance(i);
                    classDistribution[(int) inst.classValue()]++;
                }

                normalizeClassDistribution(classDistribution);
                classValue = indexWithMaxValue(classDistribution);
                classAttribute = instances.classAttribute();
            } else {
                Instances[] splitData = splitDataBasedOnAttribute(instances, splitAttribute);
                successors = new MyID3[splitAttribute.numValues()];
                for (int j = 0; j < splitAttribute.numValues(); j++) {
                    successors[j] = new MyID3();
                    successors[j].makeTree(splitData[j]);
                }
            }
        }
    }

    public Instances toNominal(Instances data) throws Exception {
        for (int n=0; n<data.numAttributes(); n++) {
            Attribute att = data.attribute(n);
            if (data.attribute(n).isNumeric()) {
                HashSet<Integer> uniqueValues = new HashSet();
                for (int i = 0; i < data.numInstances(); ++i)
                    uniqueValues.add((int) (data.instance(i).value(att)));

                List<Integer> dataValues = new ArrayList<Integer>(uniqueValues);
                dataValues.sort(new Comparator<Integer>() {
                    public int compare(Integer o1, Integer o2) {
                        if(o1 > o2){
                            return 1;
                        }else{
                            return -1;
                        }
                    }
                });

                // Search for threshold and get new Instances
                double[] infoGains = new double[dataValues.size() - 1];
                Instances[] tempInstances = new Instances[dataValues.size() - 1];
                for (int i = 0; i < dataValues.size() - 1; ++i) {
                    tempInstances[i] = setAttributeThreshold(data, att, dataValues.get(i));
                    infoGains[i] = computeIG(tempInstances[i], tempInstances[i].attribute(att.name()));
                }
                data = new Instances(tempInstances[indexWithMaxValue(infoGains)]);
            }
        }
        return data;
    }

    private static Instances setAttributeThreshold(Instances data, Attribute att, int threshold) throws Exception {
        Instances temp = new Instances(data);
        // Add thresholded attribute
        Add filter = new Add();
        filter.setAttributeName("thresholded " + att.name());
        filter.setAttributeIndex(String.valueOf(att.index() + 2));
        filter.setNominalLabels("<=" + threshold + ",>" + threshold);
        filter.setInputFormat(temp);
        Instances thresholdedData = Filter.useFilter(data, filter);

        for (int i=0; i<thresholdedData.numInstances(); i++) {
            if ((int) thresholdedData.instance(i).value(thresholdedData.attribute(att.name())) <= threshold)
                thresholdedData.instance(i).setValue(thresholdedData.attribute("thresholded " + att.name()), "<=" + threshold);
            else
                thresholdedData.instance(i).setValue(thresholdedData.attribute("thresholded " + att.name()), ">" + threshold);
        }
        thresholdedData = WekaIFace.removeAttribute(thresholdedData, String.valueOf(att.index() + 1));
        thresholdedData.renameAttribute(thresholdedData.attribute("thresholded " + att.name()), att.name());
        return thresholdedData;
    }

    @Override
    public double classifyInstance(Instance instance)
            throws NoSupportForMissingValuesException {
        System.out.println(instance);
        if (instance.hasMissingValue())
            throw new NoSupportForMissingValuesException("classifier.MyID3: This classifier can not handle missing value");
        if (splitAttribute == null)
            return classValue;
        else
            return successors[(int) instance.value(splitAttribute)].classifyInstance(instance);
    }

    @Override
    public double[] distributionForInstance(Instance instance)
            throws NoSupportForMissingValuesException {
        if (instance.hasMissingValue())
            throw new NoSupportForMissingValuesException("classifier.MyID3: Cannot handle missing values");
        if (splitAttribute == null)
            return classDistribution;
        else {
            if(splitAttribute.value(0).contains("<=")){
                int threshold = Integer.valueOf(splitAttribute.value(0).substring(2, 3));
                if(instance.value(splitAttribute) > threshold)
                    return successors[1].distributionForInstance(instance);
                else
                    return successors[0].distributionForInstance(instance);
            }
            return successors[(int) instance.value(splitAttribute)].distributionForInstance(instance);
        }
    }

    private void normalizeClassDistribution(double[] array) {
        double sum = 0;
        for (double d : array)
            sum += d;

        if (!Double.isNaN(sum) && sum != 0) {
            for (int i = 0; i < array.length; ++i)
                array[i] /= sum;
        }
    }

    private static Instances[] splitDataBasedOnAttribute(Instances instances, Attribute attribute) {
        Instances[] splittedData = new Instances[attribute.numValues()];
        for (int j = 0; j < attribute.numValues(); j++)
            splittedData[j] = new Instances(instances, instances.numInstances());

        for (int i = 0; i < instances.numInstances(); i++) {
            int attValue = (int) instances.instance(i).value(attribute);
            splittedData[attValue].add(instances.instance(i));
        }

        for (Instances currentSplitData : splittedData)
            currentSplitData.compactify();
        return splittedData;
    }

    private double computeIG(Instances instances, Attribute attribute)
            throws Exception {
        double IG = computeE(instances);
        Instances[] splitData = splitDataBasedOnAttribute(instances, attribute);
        for (int j = 0; j < attribute.numValues(); j++) {
            if (splitData[j].numInstances() > 0) {
                IG -= ((double) splitData[j].numInstances() /
                        (double) instances.numInstances()) *
                        computeE(splitData[j]);
            }
        }
        return IG;
    }

    private static double computeE(Instances instances) throws Exception {
        double[] labelCounts = new double[instances.numClasses()];
        for (int i = 0; i < instances.numInstances(); ++i)
            labelCounts[(int) instances.instance(i).classValue()]++;

        double entropy = 0;
        for (int i = 0; i < labelCounts.length; i++) {
            if (labelCounts[i] > 0) {
                double proportion = labelCounts[i] / instances.numInstances();
                entropy -= (proportion) * log2(proportion);
            }
        }
        return entropy;
    }

    private boolean equalValue(double n, double m) {
        return ((n == m) || Math.abs(n - m) < 1e-6);
    }

    protected static int indexWithMaxValue(double[] array) {
        double max = 0;
        int index = 0;

        if (array.length > 0) {
            for (int i = 0; i < array.length; ++i) {
                if (array[i] > max) {
                    max = array[i];
                    index = i;
                }
            }
            return index;
        } else {
            return -1;
        }
    }

    @Override
    public String toString() {

        if ((classDistribution == null) && (successors == null)) {
            return "classifier.MyID3: No model built yet.";
        }
        return "classifier.MyID3\n\n" + treeToString(0);
    }

    protected String treeToString(int level) {
        StringBuilder text = new StringBuilder();

        if (splitAttribute == null) {
            if (Instance.isMissingValue(classValue)) {
                text.append(": null");
            } else {
                text.append(": ").append(classAttribute.value((int) classValue));
            }
        } else {
            for (int j = 0; j < splitAttribute.numValues(); j++) {
                text.append("\n");
                for (int i = 0; i < level; i++) {
                    text.append("|  ");
                }
                text.append(splitAttribute.name()).append(" = ").append(splitAttribute.value(j));
                text.append(successors[j].treeToString(level + 1));
            }
        }
        return text.toString();
    }
}
