package AxCore.Testing

import spinal.core._
import spinal.core.sim._
import AxCore.Config
import scala.language.postfixOps
import AxCore.Testing.TestCases.{Test_SNC_W4, Test_mpFPMA, Test_SA_4x4}



object OverallFunctionalTest extends App {


  // **** Functional Test 1 ****
  // This test is for SNC (Subnormal Number Conversion)
  Test_SNC_W4.runTest()


  // **** Functional Test 2 ****
  // This test is for mpFPMA (mix-precision Floating-Point Multiplication Approximation)
  val Wq_TestValue = 2
  Test_mpFPMA.runTest(
    A_FP_TestStartValue = -100,
    A_FP_TestEndValue   = 101,
    A_FP_TestStep       = 1,
    W_FP_TestValue      = Wq_TestValue,
    Wq_FmtSel           = 0                // Wq Format Select. 00,01 for E3M0, 10 for E2M1, 11 for E1M2
  )


  // **** Functional Test 3 ****
  Test_SA_4x4.runTest()

}


