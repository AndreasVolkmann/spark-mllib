import org.apache.spark.SparkConf
import org.apache.spark.api.java.JavaRDD
import org.apache.spark.api.java.JavaSparkContext
import org.apache.spark.mllib.recommendation.ALS
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel
import org.apache.spark.mllib.recommendation.Rating
import scala.Tuple2
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

        println("Training: $numTraining, validation: $numValidation, test: $numTest")

        val ranks = listOf(8, 12)
        val lambdas = listOf(1.0, 10.0)
        val numIters = listOf(10, 20)
        var bestModel: MatrixFactorizationModel? = null
        var bestValidationRmse = Double.MAX_VALUE
        var bestRank = 0
        var bestLambda = -1.0
        var bestNumIter = -1
        for (i in 0..1) {
            val rank = ranks[i];
            val lambda = lambdas[i];
            val numIter = numIters[i];
            val model = ALS.train(training.rdd(), rank, numIter, lambda)
            val validationRmse = computeRmse(model, validation, numValidation)
            println("RMSE (validation) = $validationRmse for the model trained with rank = $rank, lambda = $lambda, and numIter = $numIter.")
            if (validationRmse < bestValidationRmse) {
                bestModel = model
                bestValidationRmse = validationRmse
                bestRank = rank
                bestLambda = lambda
                bestNumIter = numIter
            }
        }

        val testRmse = computeRmse(bestModel!!, test, numTest)

        println("The best model was trained with rank = $bestRank and lambda $bestLambda" +
                " and numIter = $bestNumIter, and its RMSE on the test set is $testRmse.")

        // create a naive baseline and compare it with the best model
        val meanRating = training.union(validation).map(Rating::rating).mean()
        val baselineRmse = Math.sqrt(test.map { (meanRating - it.rating()) * (meanRating - it.rating()) }.mean())
        val improvement = (baselineRmse - testRmse) / baselineRmse * 100
        println("The best model imrpoves the baseline by $improvement%.")

        // make personalized recommendations

        val myRatedMovieIds = myRatings.map(Rating::product).toSet()
        val candidates = sc.parallelize(movies.keys.filterNot { myRatedMovieIds.contains(it) })
        val recommendations = bestModel
                .predict(candidates.mapToPair { Tuple2(0, it) })
                .collect()
                .sortedByDescending(Rating::rating)
                .take(50)

        println("Movies recommended for you: ")
        recommendations.forEachIndexed { i, r ->
            println("$i: ${movies[r.product()]}")
        }

        sc.stop()
    }

    fun JavaRDD<Double>.mean(): Double {
        val size = this.count()
        return this.reduce { v1, v2 -> v1 + v2 } / size
    }

    fun computeRmse(model: MatrixFactorizationModel, data: JavaRDD<Rating>, n: Long): Double {
        val map = data.mapToPair { r -> Tuple2(r.user(), r.product()) }
        val predictions = model.predict(map)
        val predictionsAndRatings = predictions.mapToPair { Tuple2(Tuple2(it.user(), it.product()), it.rating()) }
                .join(data.mapToPair { Tuple2(Tuple2(it.user(), it.product()), it.rating()) })
                .values()
        return Math.sqrt(predictionsAndRatings
                .map { (it._1 - it._2) * (it._1 - it._2) }
                .reduce { v1, v2 -> v1 + v2 } / n)
    }

    fun loadRatings(path: String) = File("data/$path").readLines().map {
        val fields = it.split("::")
        toRating(fields)
    }.filter { it.rating() > 0.0 }


    fun toRating(fields: List<String>) = Rating(fields[0].toInt(), fields[1].toInt(), fields[2].toDouble())


}