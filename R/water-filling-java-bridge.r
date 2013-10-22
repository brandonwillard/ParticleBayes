
find.log.alpha <- function(rLogWeights, rLogWeightsSum=NULL, rN) {
  jSamplingUtils = J("com.statslibextensions.statistics.ExtSamplingUtils")
  logWeights = as.double(rLogWeights)
  N = as.integer(rN)

  if (is.null(rLogWeightsSum)) {
    logSum = su0$logSum(logWeights)  
  } else {
    logSum = as.double(rLogWeightsSum)
  }

  result = jSamplingUtils$findLogAlpha(logWeights, logSum, N)
  
  return(result)
}


water.filling.resample <- function(rLogWeights, rLogWeightsSum=NULL, rObjects, rN, seed=NULL) {
  jSamplingUtils = J("com.statslibextensions.statistics.ExtSamplingUtils")
  rng = new(J("java.util.Random"))
  if (!is.null(seed))
    rng$setSeed(seed)
  objects = new(J("java.util.ArrayList"))
  for (obj in rObjects) {
    objects$add(as.character(obj))
  }
      #J("com.google.common.primitives.Doubles")$asList(as.double(rObjects)))
  logWeights = as.double(rLogWeights)
  N = as.integer(rN)

  if (is.null(rLogWeightsSum)) {
    logSum = jSamplingUtils$logSum(logWeights)  
  } else {
    logSum = as.double(rLogWeightsSum)
  }

  result = jSamplingUtils$waterFillingResample(logWeights, logSum, objects, rng, N)
  
  hm = result$asMap()
  # convert to R list
  keySet = .jrcall(hm,"keySet")
  an_iter = .jrcall(keySet,"iterator")
  aList = list()
  while(.jrcall(an_iter,"hasNext")) {
      key = .jrcall(an_iter,"next", simplify=F);
      skey = as.character(.jsimplify(key))
      val = .jrcall(hm,"get",key)
      if (.jinstanceof(val, "com.statslibextensions.math.MutableDoubleCount")) {
        aList[[skey]]$count = val$getCount() 
        aList[[skey]]$value = val$getValue() 
      } else {
        aList[[skey]] = val$getValue()
      }
  }
  
  return(aList)
}

