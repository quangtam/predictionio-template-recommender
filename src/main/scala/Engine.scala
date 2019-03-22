package vn.accesstrade.recommendation

import org.apache.predictionio.controller.EngineFactory
import org.apache.predictionio.controller.Engine

case class Query(
  user: String,
  num: Int
)

case class PredictedResult(
  campaignScores: Array[CampaignScore]
)

case class ActualResult(
  ratings: Array[Rating]
)

case class CampaignScore(
  campaign: String,
  score: Double
)

object RecommendationEngine extends EngineFactory {
  def apply() = {
    new Engine(
      classOf[DataSource],
      classOf[Preparator],
      Map("als" -> classOf[ALSAlgorithm]),
      classOf[Serving])
  }
}
