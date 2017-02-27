import org.apache.spark.SparkConf
import org.apache.spark.api.java.JavaSparkContext
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel
import org.apache.spark.mllib.recommendation.Rating
import org.apache.spark.rdd.RDD
import java.io.File

object MovieLensALS {

    @JvmStatic
    fun main(args: Array<String>) {

        val conf = SparkConf()
                .setAppName("MovieLensALS")
                .setMaster("local")
                .set("spark.executor.memory", "2g")
        val sc = JavaSparkContext(conf)

        // load personal ratings
        val myRatings = loadRatings("personalRatings.txt")
        val myRatingsRDD = sc.parallelize(myRatings, 1)

        // load ratings and movie titles
        val movieLensHomeDir = "data"

        val ratings = sc.textFile(File("$movieLensHomeDir/ratings.dat").absolutePath).map { line ->
            val fields = line.split("::")
            // format: (timestamp % 10, Rating(userId, movieId, rating))
            fields[3].toLong() % 10 to toRating(fields)
        }

        val movies = sc.textFile(File("$movieLensHomeDir/movies.dat").absolutePath).map { line ->
            val fields = line.split("::")
            // format: (movieId, movieName)
            fields[0].toInt() to fields[1]
        }.collect().toMap()

        // code here
        val numRatings = ratings.count()
        val numUsers = ratings.map { it.second.user() }.distinct().count()
        val numMovies = ratings.map { it.second.product() }.distinct().count()
        println("Got $numRatings ratings from $numUsers users on $numMovies movies.")

        val numPartitions = 4
        val training = ratings.filter { it.first < 6 }
                .map { it.second }
                .union(myRatingsRDD)
                .repartition(numPartitions)
                .cache()
        val validation = ratings.filter { it.first in 6..7 }
                .map { it.second }
                .repartition(numPartitions)
                .cache()
        val test = ratings.filter { it.first >= 8 }
                .map { it.second }
                .cache()

        val numTraining = training.count()
        val numValidation = validation.count()
        val numTest = test.count()

        println("Training: $numValidation, validation: $numValidation, test: $numTest")

        sc.stop()
    }

    fun computeRmse(model: MatrixFactorizationModel, data: RDD<Rating>, n: Long): Double {
        TODO()
    }

    fun loadRatings(path: String) = File("data/$path").readLines().map {
        val fields = it.split("::")
        toRating(fields)
    }


    fun toRating(fields: List<String>) = Rating(fields[0].toInt(), fields[1].toInt(), fields[2].toDouble())


}