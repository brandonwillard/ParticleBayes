package org.bitbucket.brandonwillard.particlebayes.radapters;

import gov.sandia.cognition.math.matrix.Matrix;
import gov.sandia.cognition.math.matrix.Vector;
import gov.sandia.cognition.math.matrix.VectorFactory;
import gov.sandia.cognition.statistics.DataDistribution;

import java.util.List;

import plm.logit.fruehwirth.FruehwirthLogitPLFilter;
import plm.logit.fruehwirth.FruehwirthLogitParticle;

import com.google.common.collect.Lists;
import com.statslibextensions.util.ObservedValue;

public class FruehwirthLogitPLAdapter {
  public static List<DataDistribution<FruehwirthLogitParticle>> batchUpdate(
      FruehwirthLogitPLFilter filter, double[] y, Matrix data) {
    List<DataDistribution<FruehwirthLogitParticle>> result = Lists.newArrayList();
    DataDistribution<FruehwirthLogitParticle> currentDist = filter.createInitialLearnedObject();
    result.add(currentDist.clone());
//    double [][] weights = new double[y.length][this.numParticles];
//    double [][][] stateMeans = new double[y.length][this.numParticles][data.getNumColumns()];
    
//    weights[0] = currentDist.asMap().values()
    for (int t = 0; t < y.length; t++) {
      final ObservedValue<Vector, Matrix> obs = ObservedValue.<Vector, Matrix>create(
          t, 
          VectorFactory.getDefault().copyValues(y[t]),
          data.getSubMatrix(t, t, 0, data.getNumColumns()-1));
      filter.update(currentDist, obs);
      result.add(currentDist.clone());
    }

    return result;
  }
}
