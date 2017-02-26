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