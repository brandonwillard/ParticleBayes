package org.bitbucket.brandonwillard.particlebayes.radapters;

import gov.sandia.cognition.math.matrix.Matrix;
import gov.sandia.cognition.math.matrix.MatrixFactory;
import gov.sandia.cognition.math.matrix.Vector;
import gov.sandia.cognition.math.matrix.VectorFactory;
import gov.sandia.cognition.statistics.DataDistribution;
import gov.sandia.cognition.statistics.distribution.MultivariateGaussian;

import java.util.Collections;
import java.util.List;
import java.util.Random;

import plm.logit.fruehwirth.multi.FruehwirthMultiPLFilter;
import plm.logit.fruehwirth.multi.FruehwirthMultiParticle;

import com.google.common.base.Function;
import com.google.common.collect.Collections2;
import com.google.common.collect.Lists;
import com.google.common.primitives.Doubles;
import com.statslibextensions.util.ObservedValue;

public class FruehwirthMultiPLAdapter {

  public static class RFruehwirthMultiPLResult {

    private final double[][][] stateMeans;
    private final double[][] logWeights;

    public RFruehwirthMultiPLResult(double[][][] stateMeans, double[][] logWeights) {
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

  /**
   * @param filter
   * @param y
   * @param M number of classes in the multinomial
   * @param data
   * @return
   */
  public static RFruehwirthMultiPLResult batchUpdate(
      double[] priorMean, double[][] priorCov, 
      double[][] F, double[][] G, double[][] modelCov,
      double[] y, int N, int M, double[][] X, int seed) {

    FruehwirthMultiPLFilter filter = new FruehwirthMultiPLFilter(
        new MultivariateGaussian(
            VectorFactory.getDefault().copyArray(priorMean),
            MatrixFactory.getDefault().copyArray(priorCov)), 
        MatrixFactory.getDefault().copyArray(F), 
        MatrixFactory.getDefault().copyArray(G), 
        MatrixFactory.getDefault().copyArray(modelCov), 
        M, new Random(seed));
    filter.setNumParticles(N);

    DataDistribution<FruehwirthMultiParticle> currentDist = filter.createInitialLearnedObject();

    final Matrix data = MatrixFactory.getDefault().copyArray(X);
    double [][] logWeights = new double[y.length][filter.getNumParticles()];
    double [][][] stateMeans = new double[y.length][filter.getNumParticles()][data.getNumColumns()];
    
    logWeights[0] = Doubles.toArray(currentDist.asMap().values());
    stateMeans[0] = 
        Collections2.transform(currentDist.asMap().keySet(), 
        new Function<FruehwirthMultiParticle, double[]>() {
          @Override
          public double[] apply(FruehwirthMultiParticle input) {
            return input.getLinearState().getMean().toArray();
          }}).toArray(new double[filter.getNumParticles()][data.getNumColumns()]);

    for (int t = 0; t < y.length; t++) {
      final ObservedValue<Vector, Matrix> obs = ObservedValue.<Vector, Matrix>create(
          t, 
          VectorFactory.getDefault().copyValues(y[t]),
          data.getSubMatrix(t, t, 0, data.getNumColumns()-1));
      filter.update(currentDist, obs);

      logWeights[t] = Doubles.toArray(currentDist.asMap().values());
      stateMeans[t] = 
          Collections2.transform(currentDist.asMap().keySet(), 
          new Function<FruehwirthMultiParticle, double[]>() {
            @Override
            public double[] apply(FruehwirthMultiParticle input) {
              return input.getLinearState().getMean().toArray();
            }}).toArray(new double[filter.getNumParticles()][data.getNumColumns()]);
    }

    return new RFruehwirthMultiPLResult(stateMeans, logWeights);
  }
}
