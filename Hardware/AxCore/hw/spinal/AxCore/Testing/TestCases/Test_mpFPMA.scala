package AxCore.Testing.TestCases

import AxCore.Config
import AxCore.SystolicArray_W4.AxCore_SharedAdd_MPWq4_SA
import spinal.core.sim._
import scala.language.postfixOps
import scala.collection.mutable.ArrayBuffer
import AxCore.Basics.{FP2BinCvt, Bin2FPCvt}
// import AxCore.Testing.TestCases.TU
import AxCore.Testing.GoldenModels.mpFPMA_W4A16
import scala.math.BigDecimal


object Test_mpFPMA {

  // Run test here, or run test at OverallFunctionalTest
  def main(args: Array[String]): Unit = {

    // Important:
    // Each Wq element should be a correct value corresponding to it's value space of selected Wq Format.
    // Value Space of E3M0: (0, 0.25, 0.5, 1,      2,             4,    8, 16)
    // Value Space of E2M1: (0,       0.5, 1, 1.5, 2,      3,     4, 6       )
    // Value Space of E1M2: (0,       0.5, 1, 1.5, 2, 2.5, 3, 3.5            )
    val Wq_TestValue = -1.5
    
    runTest(
      A_FP_TestStartValue = -100,
      A_FP_TestEndValue   = 100,
      A_FP_TestStep       = 1,
      W_FP_TestValue      = Wq_TestValue,
      Wq_FmtSel           = 3                // Wq Format Select. 00,01 for E3M0, 10 for E2M1, 11 for E1M2
    )
  }

  def runTest(A_FP_TestStartValue: Double, A_FP_TestEndValue: Double, A_FP_TestStep: Double, W_FP_TestValue: Double, Wq_FmtSel: Int): Unit = {

    // * Configuration
    val ExpoWidth = 5
    val MantWidth = 10
    val TotalWidth = 1 + ExpoWidth + MantWidth

    // * Inputs
    val count = ((BigDecimal(A_FP_TestEndValue.toString) - BigDecimal(A_FP_TestStartValue.toString)) / BigDecimal(A_FP_TestStep.toString))
      .setScale(0, BigDecimal.RoundingMode.CEILING).toInt
    val A_FP_TestRange_Temp = (0 to count).map { i =>
      BigDecimal(A_FP_TestStartValue.toString) + i * BigDecimal(A_FP_TestStep.toString)
    }
    val A_FP_TestRange = A_FP_TestRange_Temp.map(_.toDouble).toArray    // Make the TestRange Array
    // println(A_FP_TestRange.mkString("Array(", ", ", ")"))
    // println(A_FP_TestRange.length)

    val R_FPBin_Buffer       = ArrayBuffer[String]()
    val R_FP_Buffer          = ArrayBuffer[Double]()
    val StochasticBit_Buffer = ArrayBuffer[Boolean]()

    var Wq_FP     = 0.0
    var A_FP      = 0.0
    var S_FP      = 0
    var WqLock    = false


    // * Testing
    Config.vcssim.compile{
      AxCore_SharedAdd_MPWq4_SA(
        TileRow=1, TileCol=1, PERow=2, PECol=2,
        QtTotalWidth=4, ExpoWidth=ExpoWidth, MantWidth=MantWidth, Integer=ExpoWidth-1, Fraction=MantWidth+2
      )
    }.doSim { dut =>
      // simulation process
      dut.clockDomain.forkStimulus(2)
      // simulation code
      for (clk <- 0 until (A_FP_TestRange.length + 20)) {
        if (clk < 9) {
          Wq_FP = W_FP_TestValue
          WqLock = false
        } else if (clk == 9) {
          WqLock = true
        } else if (clk >= 10 && clk < 10+A_FP_TestRange.length) {
          A_FP = A_FP_TestRange(clk - 10)    // Assign the values in A_FP_TestRange one by one
        } else {
          Wq_FP     = 0
          A_FP      = 0
          S_FP      = 0
          WqLock    = false
        }

        // * Apply stimulus to the DUT
        dut.io.DinTop_Wq_FP(0)(0) #= TU.DynFP4toBinBigInt(f=Wq_FP, FmtSel=Wq_FmtSel)
        dut.io.DinTop_Wq_FP(0)(1) #= 0
        dut.io.DinLft_A_FP(0)(0)  #= TU.FPtoBinBigInt(f=A_FP, ExpoWidth=ExpoWidth, MantWidth=MantWidth)
        dut.io.DinLft_A_FP(0)(1)  #= 0
        dut.io.S_FP               #= TU.FPtoBinBigInt(f=S_FP, ExpoWidth=ExpoWidth, MantWidth=MantWidth)
        dut.io.Wq_FmtSel          #= Wq_FmtSel
        dut.io.WqLock             #= WqLock
        dut.clockDomain.waitRisingEdge()    // sample on rising edge

        // * Collect the outputs (the timing must be correct)
        if (clk >= 10+1 && clk < 10+A_FP_TestRange.length+1) {
          val StochasticBit = dut.Tiles(0)(0).PEs(0)(0).SNC_MPWq4.io.StochasticBit.toBigInt
          val R_FP_BigInt = dut.Tiles(0)(0).PEs(0)(0).io.R_FP.toBigInt
          val R_FPBin = TU.BigIntToBinary(R_FP_BigInt, TotalWidth)
          val R_FP = TU.BigIntToFP(R_FP_BigInt, ExpoWidth=ExpoWidth, MantWidth=MantWidth)

          R_FPBin_Buffer += R_FPBin
          R_FP_Buffer += R_FP
          StochasticBit_Buffer += (StochasticBit == 1)
        }
      }
      sleep(50)
    }

    // * Checking & Logging
    println(s" \n=============================================================================================================")
    println(s"<<< mpFPMA Test >>>")
    println(s"This is a test for mpFPMA (mix-precision Floating-Point Multiplication Approximation) with Wq * A.")
    println(s"The DUT output will be checked with the Golden Model output.")
    println(s"In each test cycle, the Status will show whether the DUT is correct or not.")
    println(s"=============================================================================================================\n ")


    for (i <- A_FP_TestRange.indices) {
      val A_FPBin = FP2BinCvt.FloatToFPAnyBin(f=A_FP_TestRange(i), ExpoWidth=ExpoWidth, MantWidth=MantWidth)
      val A_FP_BigInt = TU.FPtoBinBigInt(f=A_FP_TestRange(i), ExpoWidth=ExpoWidth, MantWidth=MantWidth)
      val W_FP_BigInt = TU.DynFP4toBinBigInt(f=W_FP_TestValue, FmtSel=Wq_FmtSel)

      // * Golden Model for mpFPMA
      val R_FPBin_Golden = mpFPMA_W4A16.calculate(
        Wq_FP_In=W_FP_BigInt, A_FP_In=A_FP_BigInt,
        ExpoWidth=ExpoWidth, MantWidth=MantWidth,
        Wq_FmtSel=Wq_FmtSel,
        // StochasticBit=StochasticBit_Buffer(i),
        withComp=true
      )
      val R_FP_Golden = Bin2FPCvt.FPAnyBinToFloat(FPBin=R_FPBin_Golden, ExpoWidth=ExpoWidth, MantWidth=MantWidth)

      // * Exact Multiplication Result (No Approximation)
      val R_FP_Exact = W_FP_TestValue * A_FP_TestRange(i)

      // * Check for correctness before logging
      val isCorrect = (R_FP_Buffer(i) == R_FP_Golden)
      val status = if (isCorrect) "Correct" else "Incorrect"
      val statusColored = if (isCorrect) s"\u001B[32m${status}\u001B[0m" else s"\u001B[31m${status}\u001B[0m" // Green for Correct, Red for Incorrect

      // * Prepare each String for logging
      val cycleString     = f"Cycle: $i%-5d"
      val inString        = s"In: [Wq (=$W_FP_TestValue) * A (=${A_FP_TestRange(i)})]"
      val dutOutString    = s"DUT mpFPMA Out: [R ('${R_FPBin_Buffer(i)}'=${R_FP_Buffer(i)})]"
      val goldenOutString = s"Golden mpFPMA Out: [R (=$R_FP_Golden)]"
      val statusString    = s"Status: $statusColored"

      // * Logging
      println(
        f"${cycleString}%-8s | ${inString}%-32s | ${dutOutString}%-56s | ${goldenOutString}%-42s | ${statusString}"
      )
    }

    println(s" \n=============================================================================================================")
    println(s"End of <<< mpFPMA Test >>>")
    println(s"=============================================================================================================\n ")

  }
}