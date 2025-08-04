package AxCore.Testing.TestCases

import spinal.core._
import spinal.core.sim._
import AxCore.Config
import scala.language.postfixOps
import AxCore.Basics.SNC_W4
import AxCore.Testing.GoldenModels.GoldenModel_SNC_W4


object Test_SNC_W4 {

  // Run test here, or run test at OverallFunctionalTest
  def main(args: Array[String]): Unit = {
    runTest()
  }

  def runTest(): Unit = {
    // * Inputs
    var StochasticBit = false
    var Wq_FmtSel     = 0
    var Wq_FP_In      = 0


    // * Testing
    Config.vcssim.compile{SNC_W4()}.doSim { dut =>
      // simulation process
      dut.clockDomain.forkStimulus(2)
      // simulation code

      println(s" \n=============================================================================================================")
      println(s"<<< SNC Test >>>")
      println(s"This is a test for SNC (Subnormal Number Conversion) with all possible Wq inputs.")
      println(s"In each test cycle, the Status will show whether the DUT is correct or not.")
      println(s"=============================================================================================================\n ")

      for (clk <- 0 until 100) {
        // Traverse all possible inputs
        if (clk >= 10 && clk < 10+64) {
          StochasticBit = (clk % 2 == 0)     // Simulate randomness
          Wq_FmtSel     = (clk - 10) / 16    // 00,01 for E3M0, 10 for E2M1, 11 for E1M2. Only valid in W4.
          Wq_FP_In      = (clk - 10) % 16
        } else {
          StochasticBit = false
          Wq_FmtSel     = 0
          Wq_FP_In      = 0
        }

        // * DUT
        // Input
        dut.io.StochasticBit #= StochasticBit
        dut.io.Wq_FmtSel     #= Wq_FmtSel
        dut.io.Wq_FP_In      #= Wq_FP_In
        dut.clockDomain.waitRisingEdge()    // sample on rising edge
        // Output
        val dut_Wq_FP_Out = dut.io.Wq_FP_Out.toBigInt
        val dut_Wq_NotZero = dut.io.Wq_NotZero.toBoolean

        // * Golden Model
        val Expected = GoldenModel_SNC_W4.calculate(
          Wq_FP_In      = Wq_FP_In,
          Wq_FmtSel     = Wq_FmtSel,
          StochasticBit = StochasticBit
        )

        // * Check for correctness before logging
        val isCorrect = (dut_Wq_FP_Out == Expected.Wq_FP_Out) && (dut_Wq_NotZero == Expected.Wq_NotZero)
        val status = if (isCorrect) "Correct" else "Incorrect"
        val statusColored = if (isCorrect) s"\u001B[32m${status}\u001B[0m" else s"\u001B[31m${status}\u001B[0m" // Green for Correct, Red for Incorrect

        // * Logging
        if (clk >= 10 && clk < 10+64) {
          println(
            f"Cycle: ${clk-10}%-3d | " +
              f"In: [FmtSel=${TU.BigIntToBinary(Wq_FmtSel, 2)}, Wq_FP4_In=${TU.BigIntToBinary(Wq_FP_In, 4)}, Stoc=${if(StochasticBit) 1 else 0}] | " +
              f"DUT Out: [Wq_E3M2_Out=${TU.BigIntToBinary(dut_Wq_FP_Out, 6)}, NotZero=${if(dut_Wq_NotZero) 1 else 0}] | " +
              f"Golden: [Wq_E3M2_Out=${TU.BigIntToBinary(Expected.Wq_FP_Out, 6)}, NotZero=${if(Expected.Wq_NotZero) 1 else 0}] | " +
              f"Status: $statusColored%-18s"
          )
        }

      }
      sleep(50)

      println(s" \n=============================================================================================================")
      println(s"End of <<< SNC Test >>>")
      println(s"=============================================================================================================\n ")
    }
  }
}