package AxCore.Testing.TestCases

import spinal.core._
import spinal.core.sim._
import AxCore.Config
import scala.language.postfixOps
import AxCore.SystolicArray_W4.AxCore_SharedAdd_MPWq4_SA
import AxCore.Basics.{FP2BinCvt, Bin2FPCvt}
import AxCore.Testing.GoldenModels.SA_4x4


object Test_SA_4x4 {

  // Run test here, or run test at OverallFunctionalTest
  def main(args: Array[String]): Unit = {
    runTest()
  }

  def runTest(): Unit = {

    // * Configuration
    val ExpoWidth = 5
    val MantWidth = 10
    val Integer = ExpoWidth - 1
    val Fraction = MantWidth + 2

    // * Inputs
    // Random
    // val Wq_Matrix = TU.genRandomDynFP4Matrix(NumRows=4, NumCols=4, FmtSel=2)

    val Wq_FmtSel = 0    // Wq Format Select. 00,01 for E3M0, 10 for E2M1, 11 for E1M2

    val Wq_Matrix = Array(
      Array(1.0, 1.0, 1.0, 1.0),
      Array(2.0, 2.0, 2.0, 2.0),
      Array(4.0, 4.0, 4.0, 4.0),
      Array(8.0, 8.0, 8.0, 8.0),
    )

    val A_Matrix = Array(
      Array(10.0, 10.0, 10.0, 10.0),
      Array(20.0, 20.0, 20.0, 20.0),
      Array(30.0, 30.0, 30.0, 30.0),
      Array(40.0, 40.0, 40.0, 40.0),
    )


    var S_FP   = 0
    var WqLock = false

    var Wq_FP_c0 = 0.0
    var Wq_FP_c1 = 0.0
    var Wq_FP_c2 = 0.0
    var Wq_FP_c3 = 0.0

    var A_FP_r0 = 0.0
    var A_FP_r1 = 0.0
    var A_FP_r2 = 0.0
    var A_FP_r3 = 0.0

    // val O_UnNorm_Matrix  = Array.ofDim[Double](4, 4)
    // val O_UnScale_Matrix = Array.ofDim[Double](4, 4)
    val Oq_Matrix = Array.ofDim[Double](4, 4)

    // * Testing
    Config.vcssim.compile{
      AxCore_SharedAdd_MPWq4_SA(
        TileRow=1, TileCol=1, PERow=4, PECol=4,
        QtTotalWidth=4, ExpoWidth=ExpoWidth, MantWidth=MantWidth, Integer=Integer, Fraction=Fraction
      )
    }.doSim { dut =>
      // simulation process
      dut.clockDomain.forkStimulus(2)
      // simulation code
      for (clk <- 0 until 100) {
        // Wq Format as E3M0
        if (clk >= 5 && clk < 9) {
          val WqIdx = 8 - clk
          Wq_FP_c0 = Wq_Matrix(WqIdx)(0)
          Wq_FP_c1 = Wq_Matrix(WqIdx)(1)
          Wq_FP_c2 = Wq_Matrix(WqIdx)(2)
          Wq_FP_c3 = Wq_Matrix(WqIdx)(3)
          WqLock = false
        } else if (clk >= 9 && clk < 10) {
          WqLock = true
          Wq_FP_c0 = 0.0
          Wq_FP_c1 = 0.0
          Wq_FP_c2 = 0.0
          Wq_FP_c3 = 0.0
        } else if (clk >= 10 && clk < 14) {
          val AIdx = clk - 10
          A_FP_r0 = A_Matrix(AIdx)(0)
          A_FP_r1 = A_Matrix(AIdx)(1)
          A_FP_r2 = A_Matrix(AIdx)(2)
          A_FP_r3 = A_Matrix(AIdx)(3)
          S_FP   = 1
          WqLock = true
        } else {
          Wq_FP_c0 = 0.0
          Wq_FP_c1 = 0.0
          Wq_FP_c2 = 0.0
          Wq_FP_c3 = 0.0
          A_FP_r0  = 0.0
          A_FP_r1  = 0.0
          A_FP_r2  = 0.0
          A_FP_r3  = 0.0
          S_FP   = 1
          WqLock = false
        }

        // * DUT
        dut.io.DinTop_Wq_FP(0)(0) #= TU.DynFP4toBinBigInt(f=Wq_FP_c0, FmtSel=Wq_FmtSel)
        dut.io.DinTop_Wq_FP(0)(1) #= TU.DynFP4toBinBigInt(f=Wq_FP_c1, FmtSel=Wq_FmtSel)
        dut.io.DinTop_Wq_FP(0)(2) #= TU.DynFP4toBinBigInt(f=Wq_FP_c2, FmtSel=Wq_FmtSel)
        dut.io.DinTop_Wq_FP(0)(3) #= TU.DynFP4toBinBigInt(f=Wq_FP_c3, FmtSel=Wq_FmtSel)
        dut.io.DinLft_A_FP(0)(0)  #= TU.FPtoBinBigInt(f=A_FP_r0, ExpoWidth=ExpoWidth, MantWidth=MantWidth)
        dut.io.DinLft_A_FP(0)(1)  #= TU.FPtoBinBigInt(f=A_FP_r1, ExpoWidth=ExpoWidth, MantWidth=MantWidth)
        dut.io.DinLft_A_FP(0)(2)  #= TU.FPtoBinBigInt(f=A_FP_r2, ExpoWidth=ExpoWidth, MantWidth=MantWidth)
        dut.io.DinLft_A_FP(0)(3)  #= TU.FPtoBinBigInt(f=A_FP_r3, ExpoWidth=ExpoWidth, MantWidth=MantWidth)
        dut.io.S_FP               #= TU.FPtoBinBigInt(f=S_FP, ExpoWidth=ExpoWidth, MantWidth=MantWidth)
        dut.io.Wq_FmtSel          #= Wq_FmtSel
        dut.io.WqLock             #= WqLock
        dut.clockDomain.waitRisingEdge()    // sample on rising edge

        // * Collect the outputs (the timing must be correct)
        if (clk >= 12 && clk < 16) {
          val r = clk - 12
          for (c <- 0 until 4) {
            // Result After Norm
            val ResultBigInt = dut.NormGroup(0)(c).io.result.toBigInt
            Oq_Matrix(r)(c) = TU.BigIntToFP(BI=ResultBigInt, ExpoWidth=ExpoWidth, MantWidth=MantWidth)
          }
        }


      }
      sleep(50)
    }


    // * Checking & Logging
    println(s" \n===============================================================================================")
    println(s"<<< AxCore Systolic Array Test >>>")
    println(s"In this test, we conduct a 4x4 Approximate MatMul: Oq = A * Wq.")
    println(s"===============================================================================================")


    println(s" \nA Matrix =")
    for (row <- A_Matrix) {
      println(row.map(elem => f"$elem%7.1f").mkString("  "))
    }

    println(s" \nWq Matrix =")
    for (row <- Wq_Matrix) {
      println(row.map(elem => f"$elem%7.1f").mkString("  "))
    }

    println(s" \nOq Matrix from DUT (The output of Approximate MatMul) =")
    for (row <- Oq_Matrix) {
      println(row.map(elem => f"$elem%7.1f").mkString("  "))
    }

    // * Golden Model of SA with mpFPMA
    val O_UnScale_Matrix_Golden = SA_4x4.calculate(A_Matrix=A_Matrix, Wq_Matrix=Wq_Matrix, ExpoWidth=ExpoWidth, MantWidth=MantWidth, Wq_FmtSel=Wq_FmtSel)
    println(s" \nOq Matrix from Golden (The output of Exact MatMul) =")
    for (row <- O_UnScale_Matrix_Golden) {
      println(row.map(elem => f"$elem%7.1f").mkString("  "))
    }

    println(s" \n=============================================================================================================")
    println(s"End of <<< AxCore Systolic Array Test >>>")
    println(s"=============================================================================================================\n ")
  }
}