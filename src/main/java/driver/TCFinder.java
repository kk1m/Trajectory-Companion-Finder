package driver;

import Utils.Cli;
import com.datastax.spark.connector.japi.CassandraRow;
import geometry.TCPoint;
import geometry.TCRegion;
import mapReduce.*;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.lang.StringUtils;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import scala.Tuple2;

import java.util.List;

import static com.datastax.spark.connector.japi.CassandraJavaUtil.javaFunctions;

public class TCFinder
{
    private static String inputFilePath = "";
    private static String outputDir = "";
    private static double distanceThreshold = 0.0001;
    private static int densityThreshold = 3;
    private static int timeInterval = 50;
    private static int durationThreshold = 3;
    private static int numSubPartitions = 2;
    private static boolean debugMode = false;
    private static boolean useCassandraInput = true;

    public static void main( String[] args ) {
        Cli parser = new Cli(args);
        parser.parse();

        if (parser.getCmd() == null)
            System.exit(0);

        initParams(parser);

        SparkConf sparkConf = new SparkConf()
                .setAppName("TrajectoryCompanionFinder")
                .set("spark.cassandra.connection.host", "127.0.0.1"); //

        // force to local mode if it is debug
        if (debugMode) sparkConf.setMaster("local[*]");

        //TODO abstract this block into its own class e.g. getInputFromCassandra(hostname)
        JavaRDD<String> input;
        if (useCassandraInput) {

            //Create JavaRDD from cassandraTable
            JavaSparkContext sc = new JavaSparkContext(sparkConf);

            //TODO remove hardcoded cassandra table references and use parameters instead
            JavaRDD<String> cassandraRowsRDD = javaFunctions(sc).cassandraTable("spark", "tcf_data")
                    .select("objectid", "latitude", "longitude", "timestamp")
                    .map(new Function<CassandraRow, String>() {
                        public String call(CassandraRow cassandraRow) throws Exception {
                            return cassandraRow.fieldValues()
                                    .toString()
                                    .replace("Vector(", "")
                                    .replace(")", "")
                                    .replace(" ", "");
                        }
                    });

            System.out.println("CassandraRowsRDD.count: " + cassandraRowsRDD.count());
            System.out.println("Data as CassandraRows: \n" + StringUtils.join(cassandraRowsRDD.collect(), "\n"));
            input = cassandraRowsRDD;

        } else {
            JavaSparkContext ctx = new JavaSparkContext(sparkConf);
            //JavaRDD<String> file = ctx.textFile(inputFilePath, 100);
            input = ctx.textFile(inputFilePath, 100);

            System.out.println("file.count: " + input.count() + "\n");
            System.out.println("Data as CassandraRows: \n" + StringUtils.join(input.collect(), "\n"));
        }
        // partition the entire data set into trajectory slots
        JavaPairRDD<Integer, Iterable<TCPoint>> slotsRDD =
                input.mapToPair(new TrajectorySlotMapper(timeInterval)).groupByKey();

        // partition each slot into sub partition
        JavaRDD<Tuple2<Integer, TCRegion>> subPartitionsRDD =
                slotsRDD.flatMap(new KDTreeSubPartitionMapper(numSubPartitions));

        // find coverage density connection in each sub partition
        // merge coverage density connection per slot
        JavaPairRDD<Integer, List<Tuple2<Integer, Integer>>> densityConnectionRDD =
                subPartitionsRDD.mapToPair(new CoverageDensityConnectionMapper(densityThreshold))
                        .reduceByKey(new CoverageDensityConnectionReducer());

        // invert indexes base on the density connection found. such that
        // the key is the accompanied object pair; the value is the slot id
        JavaPairRDD<String, Integer> densityConnectionInvertedIndexRDD = densityConnectionRDD.
                flatMapToPair(new CoverageDensityConnectionInvertedIndexer()).distinct();
        JavaPairRDD<String, Iterable<Integer>> TCMapRDD
                = densityConnectionInvertedIndexRDD.groupByKey();

        // find continuous trajectory companions
        JavaPairRDD<String, Iterable<Integer>> resultRDD =
                TCMapRDD.filter(new TrajectoryCompanionFilter(durationThreshold));

        System.out.println(String.format("Saving result to %s", outputDir));
        resultRDD.saveAsTextFile(outputDir);
        resultRDD.take(1);

        //ctx.stop();


    }

    private static void initParams(Cli parser)
    {
        String foundStr = Cli.ANSI_GREEN + "param -%s is set. Use custom value: %s" + Cli.ANSI_RESET;
        String notFoundStr = Cli.ANSI_RED + "param -%s not found. Use default value: %s" + Cli.ANSI_RESET;
        CommandLine cmd = parser.getCmd();

        try {

            // input
            if (cmd.hasOption(Cli.OPT_STR_INPUTFILE)) {
                inputFilePath = cmd.getOptionValue(Cli.OPT_STR_INPUTFILE);
            } else {
                System.err.println("Input file not defined. Aborting...");
                parser.help();
            }

            // output
            if (cmd.hasOption(Cli.OPT_STR_OUTPUTDIR)) {
                outputDir = cmd.getOptionValue(Cli.OPT_STR_OUTPUTDIR);
            } else {
                System.err.println("Output directory not defined. Aborting...");
                parser.help();
            }

            // debug
            if (cmd.hasOption(Cli.OPT_STR_DEBUG)) {
                debugMode = true;
                System.out.println("Enter debug mode. master forces to be local");
            }

            // distance threshold
            if (cmd.hasOption(Cli.OPT_STR_DISTTHRESHOLD)) {
                distanceThreshold = Double.parseDouble(cmd.getOptionValue(Cli.OPT_STR_DISTTHRESHOLD));
                System.out.println(String.format(foundStr,
                        Cli.OPT_STR_DISTTHRESHOLD, distanceThreshold));
            } else {
                System.out.println(String.format(notFoundStr,
                        Cli.OPT_STR_DISTTHRESHOLD, distanceThreshold));
            }

            // density threshold
            if (cmd.hasOption(Cli.OPT_STR_DENTHRESHOLD)) {
                densityThreshold = Integer.parseInt(cmd.getOptionValue(Cli.OPT_STR_DENTHRESHOLD));
                System.out.println(String.format(foundStr,
                        Cli.OPT_STR_DENTHRESHOLD, densityThreshold));
            } else {
                System.out.println(String.format(notFoundStr,
                        Cli.OPT_STR_DENTHRESHOLD, densityThreshold));
            }

            // time interval
            if (cmd.hasOption(Cli.OPT_STR_TIMEINTERVAL)) {
                timeInterval = Integer.parseInt(cmd.getOptionValue(Cli.OPT_STR_TIMEINTERVAL));
                System.out.println(String.format(foundStr,
                        Cli.OPT_STR_TIMEINTERVAL, timeInterval));
            } else {
                System.out.println(String.format(notFoundStr,
                        Cli.OPT_STR_TIMEINTERVAL, timeInterval));
            }

            // life time
            if (cmd.hasOption(Cli.OPT_STR_LIFETIME)) {
                durationThreshold = Integer.parseInt(cmd.getOptionValue(Cli.OPT_STR_LIFETIME));
                System.out.println(String.format(foundStr,
                        Cli.OPT_STR_LIFETIME, durationThreshold));
            } else {
                System.out.println(String.format(notFoundStr,
                        Cli.OPT_STR_LIFETIME, durationThreshold));
            }

            // number of  sub-partitions
            if (cmd.hasOption(Cli.OPT_STR_NUMPART)) {
                numSubPartitions = Integer.parseInt(cmd.getOptionValue(Cli.OPT_STR_NUMPART));
                System.out.println(String.format(foundStr,
                        Cli.OPT_STR_NUMPART, numSubPartitions));
            } else {
                System.out.println(String.format(notFoundStr,
                        Cli.OPT_STR_NUMPART, numSubPartitions));
            }
        }
        catch(NumberFormatException e) {
            System.err.println(String.format("Error parsing argument. Exception: %s", e.getMessage()));
            System.exit(0);
        }
    }
}
