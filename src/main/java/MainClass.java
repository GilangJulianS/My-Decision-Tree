import support.WekaIFace;
import weka.classifiers.Classifier;
import weka.core.Instances;

/**
 * Created by gilang on 29/09/2015.
 */
public class MainClass {

    public static final int MODE_TRAIN = 0;
    public static final int MODE_CLASSIFY = 1;
    public static final int MODE_CROSS_VALIDATE = 2;
    public static final int MODE_EVALUATE_SPLIT = 3;
    public static final int MODE_VALIDATE_TEST_SET = 4;



    public static void main(String[] args){
        String dataFile = null;
        String testFile = null;
        String modelName;
        boolean prune = false;
        Classifier classifier = null;
        int algorithm = -1;
        int mode = -1;


        for(int i=0; i<args.length; i+=2){
            if(args[i].equals("-data")){
                dataFile = args[i+1];
            }else if(args[i].equals("-al")){
                if(args[i+1].equals("bayes")){
                    algorithm = WekaIFace.BAYES;
                }else if(args[i+1].equals("id3")){
                    algorithm = WekaIFace.ID3;
                }else if(args[i+1].equals("j48")){
                    algorithm = WekaIFace.J48;
                }else if(args[i+1].equals("mybayes")){
                    algorithm = WekaIFace.MY_BAYES;
                }else if(args[i+1].equals("myid3")){
                    algorithm = WekaIFace.MY_ID3;
                }else if(args[i+1].equals("myj48")){
                    algorithm = WekaIFace.MY_J48;
                }else if(args[i+1].equals("ann")){
                    algorithm = WekaIFace.MY_ANN;
                }
            }else if(args[i].equals("-mode")){
                if(args[i+1].equals("train")){
                    mode = MainClass.MODE_TRAIN;
                }else if(args[i+1].equals("classify")){
                    mode = MainClass.MODE_CLASSIFY;
                }else if(args[i+1].equals("crossvalidate")){
                    mode = MainClass.MODE_CROSS_VALIDATE;
                }else if(args[i+1].equals("evaluatesplit")){
                    mode = MainClass.MODE_EVALUATE_SPLIT;
                }
            }else if(args[i].equals("-test")){
                testFile = args[i+1];
            }else if(args[i].equals("-model")){
                try {
                    classifier = WekaIFace.loadModel(args[i + 1]);
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }else if(args[i].equals("-prune")){
                prune = true;
            }
        }
        try {
            if (mode == MainClass.MODE_TRAIN) {
                Instances data = WekaIFace.readArff(dataFile);
                classifier = WekaIFace.buildClassifier(data, algorithm, prune);
                WekaIFace.saveModel(dataFile.replace(".arff", ".model"), classifier);
            } else if (mode == MainClass.MODE_CLASSIFY) {
                Instances dataTest = WekaIFace.readArff(testFile);
                Instances classifiedInstances = WekaIFace.classifyInstances(classifier, dataTest);
            } else if (mode == MainClass.MODE_CROSS_VALIDATE) {
                Instances data = WekaIFace.readArff(dataFile);
                WekaIFace.crossValidate(data, classifier);
            }else if(mode == MainClass.MODE_EVALUATE_SPLIT){
                Instances data = WekaIFace.readArff(dataFile);
                WekaIFace.evaluateSplit(data, classifier, 30);
            }

//            debugMode();
        }catch(Exception e){
            e.printStackTrace();
        }
    }

    public static void debugMode() throws Exception {
        Instances data = WekaIFace.readArff("data/cpu.arff");
        Classifier classifier = WekaIFace.buildClassifier(data, WekaIFace.MY_J48, false);
        WekaIFace.crossValidate(data, classifier);
    }
}
