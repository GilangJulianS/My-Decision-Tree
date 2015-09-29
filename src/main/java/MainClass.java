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
        Classifier classifier = null;
        int algorithm = -1;
        int mode = -1;


        for(int i=0; i<args.length; i+=2){
            if(args[i].equals("-data")){
                dataFile = args[i+1];
            }else if(args[i].equals("-al")){
                if(args[i+1].equals("bayes")){
                    algorithm = WekaHelper.BAYES;
                }else if(args[i+1].equals("id3")){
                    algorithm = WekaHelper.ID3;
                }else if(args[i+1].equals("j48")){
                    algorithm = WekaHelper.J48;
                }else if(args[i+1].equals("mybayes")){
                    algorithm = WekaHelper.MY_BAYES;
                }else if(args[i+1].equals("myid3")){
                    algorithm = WekaHelper.MY_ID3;
                }else if(args[i+1].equals("myj48")){
                    algorithm = WekaHelper.MY_J48;
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
                    classifier = WekaHelper.loadModel(args[i+1]);
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        }
        try {
            if (mode == MainClass.MODE_TRAIN) {
                Instances data = WekaHelper.readArff(dataFile);
                classifier = WekaHelper.buildClassifier(data, algorithm);
                WekaHelper.saveModel(dataFile.replace(".arff", ".model"), classifier);
            } else if (mode == MainClass.MODE_CLASSIFY) {
                Instances dataTest = WekaHelper.readArff(testFile);
                Instances classifiedInstances = WekaHelper.classifyInstances(classifier, dataTest);
            } else if (mode == MainClass.MODE_CROSS_VALIDATE) {
                Instances data = WekaHelper.readArff(dataFile);
                WekaHelper.crossValidate(data, classifier);
            }else if(mode == MainClass.MODE_EVALUATE_SPLIT){
                Instances data = WekaHelper.readArff(dataFile);
                WekaHelper.evaluateSplit(data, classifier, 30);
            }

            debugMode();
        }catch(Exception e){
            e.printStackTrace();
        }
    }

    public static void debugMode() throws Exception {
        Instances data = WekaHelper.readArff("data/iris.arff");
        Classifier classifier = WekaHelper.buildClassifier(data, WekaHelper.MY_J48);
        WekaHelper.crossValidate(data, classifier);
    }
}
