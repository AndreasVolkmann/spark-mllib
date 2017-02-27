import org.apache.spark.mllib.recommendation.MatrixFactorizationModel
import org.apache.spark.mllib.recommendation.Rating
import org.apache.spark.rdd.RDD


object ALS {

    fun train(ratings: RDD<Rating>, rank: Int, iterations: Int, lambda: Double): MatrixFactorizationModel {
        TODO()
    }
}