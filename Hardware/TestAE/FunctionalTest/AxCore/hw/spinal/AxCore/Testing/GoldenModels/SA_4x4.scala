package AxCore.Testing.GoldenModels

import spinal.core._
import spinal.core.sim._
import AxCore.Config
import scala.language.postfixOps
import scala.math.pow
import AxCore.Basics.{FP2BinCvt, Bin2FPCvt}
import AxCore.Testing.TestCases.TU
// import AxCore.Testing.GoldenModels.mpFPMA_W4A16


object SA_4x4 {
  def calculate(
                 A_Matrix  : Array[Array[Double]],
                 Wq_Matrix : Array[Array[Double]],
                 ExpoWidth : Int,
                 MantWidth : Int,
                 Wq_FmtSel : Int
               ): Array[Array[Double]] = {

    // * Check empty
    if (A_Matrix.isEmpty || A_Matrix(0).isEmpty || Wq_Matrix.isEmpty || Wq_Matrix(0).isEmpty) {
      throw new IllegalArgumentException("Matrix can not be empty.")
    }

    // * Size of A_Matrix = M * K
    val M = A_Matrix.length // M rows
    val K = A_Matrix(0).length // K cols

    // * Size of Wq_Matrix = K * N
    val K_Wq = Wq_Matrix.length // K rows
    val N = Wq_Matrix(0).length // N cols

    // * Check K
    if (K != K_Wq) {
      throw new IllegalArgumentException(s"Matrix size do not match. Cols of A ($K) != Rows of Wq ($K_Wq)")
    }

    val O_Matrix = Array.ofDim[Double](M, N)

    // * Matrix Multiplication with mpFPMA
    for (i <- 0 until M) {
      for (j <- 0 until N) {
        var Sum = 0.0
        for (k <- 0 until K) {
          // * Get Corresponding values
          // val A_FP_BigInt = TU.FPtoBinBigInt(f = A_Matrix(i)(k), ExpoWidth = ExpoWidth, MantWidth = MantWidth)
          // val W_FP_BigInt = TU.DynFP4toBinBigInt(f = Wq_Matrix(k)(j), FmtSel = Wq_FmtSel)

          // // * Golden Model for mpFPMA: A_Matrix(i)(k) * Wq_Matrix(k)(j)
          // val R_FPBin_Golden = mpFPMA_W4A16.calculate(
          //   Wq_FP_In = W_FP_BigInt, A_FP_In = A_FP_BigInt,
          //   ExpoWidth = ExpoWidth, MantWidth = MantWidth,
          //   Wq_FmtSel = Wq_FmtSel,
          //   withComp = true
          // )
          // val R_FP_Golden = Bin2FPCvt.FPAnyBinToFloat(FPBin = R_FPBin_Golden, ExpoWidth = ExpoWidth, MantWidth = MantWidth)

          // * Accumulate
          Sum += A_Matrix(i)(k) * Wq_Matrix(k)(j)
          // Sum += R_FP_Golden
        }
        O_Matrix(i)(j) = Sum
      }
    }

    O_Matrix
  }
}
