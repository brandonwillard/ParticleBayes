package org.bitbucket.brandonwillard.particlebayes.radapters;

import gov.sandia.cognition.math.matrix.Matrix;
import gov.sandia.cognition.math.matrix.MatrixFactory;
import gov.sandia.cognition.math.matrix.Vector;
import gov.sandia.cognition.math.matrix.VectorFactory;
import gov.sandia.cognition.statistics.DataDistribution;
import gov.sandia.cognition.statistics.distribution.MultivariateGaussian;

import java.util.Collections;
import java.util.List;
import java.util.Map.Entry;
import java.util.Random;

import plm.logit.fruehwirth.FruehwirthLogitPLFilter;
import plm.logit.fruehwirth.FruehwirthLogitParticle;

import com.google.common.base.Function;
import com.google.common.base.Preconditions;
import com.google.common.collect.Collections2;
import com.google.common.collect.Lists;
import com.google.common.primitives.Doubles;
import com.statslibextensions.math.MutableDoubleCount;
import com.statslibextensions.statistics.ExtSamplingUtils;
import com.statslibextensions.util.ExtStatisticsUtils;
import com.statslibextensions.util.ObservedValue;

public class FruehwirthLogitPLAdapter {

  public static class RFruehwirthLogitPLResult {

    private final double[][][] stateMeans;
    private final double[][] logWeights;

    public RFruehwirthLogitPLResult(double[][][] stateMeans, double[][] logWeights) {
      this.stateMeans = stateMeans;
      this.logWeights = logWeights;
    }

    public double[][][] getStateMeans() {
      return this.stateMeans;
    }

    public double[][] getLogWeights() {
      return this.logWeights;
    }
    
  }


  public static RFruehwirthLogitPLResult batchUpdate(
      double[] priorMean, double[][] priorCov, 
      double[][] F, double[][] G, double[][] modelCov,
      double[] y, int N, double[][] X, 
      boolean waterFilling, int seed) {

    FruehwirthLogitPLFilter filter = new FruehwirthLogitPLFilter(
        new MultivariateGaussian(
            VectorFactory.getDefault().copyArray(priorMean),
            MatrixFactory.getDefault().copyArray(priorCov)), 
        MatrixFactory.getDefault().copyArray(F), 
        MatrixFactory.getDefault().copyArray(G), 
        MatrixFactory.getDefault().copyArray(modelCov), 
        waterFilling,
        new Random(seed));
    filter.setNumParticles(N);

    final Matrix data = MatrixFactory.getDefault().copyArray(X);

    DataDistribution<FruehwirthLogitParticle> currentDist = filter.createInitialLearnedObject();
    double [][] logWeights = new double[y.length][filter.getNumParticles()];
    double [][][] stateMeans = new double[y.length][filter.getNumParticles()][data.getNumColumns()];
    
    logWeights[0] = Doubles.toArray(currentDist.asMap().values());
    stateMeans[0] = 
        Collections2.transform(currentDist.asMap().keySet(), 
        new Function<FruehwirthLogitParticle, double[]>() {
          @Override
          public double[] apply(FruehwirthLogitParticle input) {
            return input.getLinearState().getMean().toArray();
          }}).toArray(new double[filter.getNumParticles()][data.getNumColumns()]);

    for (int t = 0; t < y.length; t++) {
      final ObservedValue<Vector, Matrix> obs = ObservedValue.<Vector, Matrix>create(
          t, 
          VectorFactory.getDefault().copyValues(y[t]),
          data.getSubMatrix(t, t, 0, data.getNumColumns()-1));
      filter.update(currentDist, obs);


      logWeights[t] = new double[filter.getNumParticles()];
      stateMeans[t] = new double[filter.getNumParticles()][data.getNumColumns()];
      int k = 0;
      for (Entry<FruehwirthLogitParticle, ? extends Number> entry : currentDist.asMap().entrySet()) {
        final int count;
        if (entry.getValue() instanceof MutableDoubleCount) {
          count = ((MutableDoubleCount)entry.getValue()).count;
        } else {
          count = 1;
        }
        for (int r = 0; r < count; r++) {
          // FIXME might not be log scale
          final double normLogWeight = entry.getValue().doubleValue() - Math.log(count) - currentDist.getTotal();
          Preconditions.checkState(normLogWeight < 0d);
          logWeights[t][k] = normLogWeight;
          stateMeans[t][k] = entry.getKey().getLinearState().getMean().toArray();
          k++;
        }
      }
    }

    return new RFruehwirthLogitPLResult(stateMeans, logWeights);
  }
}
