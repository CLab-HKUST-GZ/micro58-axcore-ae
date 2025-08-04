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
  // Important:
  // Each Wq element should be a correct value corresponding to it's value space of selected Wq Format.
  // Value Space of E3M0: (0, 0.25, 0.5, 1,      2,             4,    8, 16)
  // Value Space of E2M1: (0,       0.5, 1, 1.5, 2,      3,     4, 6       )
  // Value Space of E1M2: (0,       0.5, 1, 1.5, 2, 2.5, 3, 3.5            )
  val Wq_TestValue = 2
  Test_mpFPMA.runTest(
    A_FP_TestStartValue = -100,
    A_FP_TestEndValue   = 101,
    A_FP_TestStep       = 1,
    W_FP_TestValue      = Wq_TestValue,
    Wq_FmtSel           = 0                // Wq Format Select. 00,01 for E3M0, 10 for E2M1, 11 for E1M2
  )


  // **** Functional Test 3 ****
  // This test is for Systolic Array
  Test_SA_4x4.runTest()

}


