package AxCore.SystolicArray_W8

import spinal.core._
import spinal.core.sim._
import AxCore.Config
import scala.language.postfixOps
import AxCore.Operators.AdderInt
import AxCore.Basics.{GuardAW, SNC_W8}


// MARK: AxCore Two-Level Spatial Array, with Weight Stationary, with Shared Adder -- PE
case class AxCore_SharedAdd_W8_PE(
                                   QtExpoWidth : Int = 4,
                                   QtMantWidth : Int = 3,
                                   ExpoWidth   : Int,
                                   MantWidth   : Int,
                                 ) extends Component {

  val TotalWidth   = 1 + ExpoWidth   + MantWidth
  val QtTotalWidth = 1 + QtExpoWidth + QtMantWidth

  val io = new Bundle {
    val Wq_CIN_FP  = in  Bits(QtTotalWidth bits)          // Wq Cascade In
    val Wq_COUT_FP = out Bits(QtTotalWidth bits)          // Wq Cascade Out

    val T_CIN_TC   = in  Bits(TotalWidth bits)            // T Cascade In (in Two's Complement)
    val T_COUT_TC  = out Bits(TotalWidth bits)            // T Cascade Out (in Two's Complement)
    val A_Vld_CIN  = in  Bool()                           // Validness of A_FP Cascade In
    val A_Vld_COUT = out Bool()                           // Validness of A_FP Cascade Out

    val R_FP       = out Bits(TotalWidth bits)            // R = Wq * A

    val WqLock     = in  Bool()                           // Locking the preloaded Wq
  }
  noIoPrefix()


  // * Weight Stationary
  val WqLockReg = Reg(Bits(QtTotalWidth bits)).init(B(0))

  when(io.WqLock) {
    WqLockReg := WqLockReg
  } otherwise {
    WqLockReg := io.Wq_CIN_FP
  }


  // * Wq Guarding (Subnormal substitution, NotZero checking)
  assert(QtExpoWidth==4 && QtMantWidth==3)
  val SNC_Wq8 = new SNC_W8()
  SNC_Wq8.io.Wq_FP_In := WqLockReg
  SNC_Wq8.io.StochasticBit := io.T_CIN_TC(MantWidth-1)
  val WqGuarded = SNC_Wq8.io.Wq_FP_Out


  // * Extend the Wq from quantized FP to bigger FP (e.g. FP4 to FP16)
  val Wq_Extend_FP = WqGuarded(QtTotalWidth-1) ## B(0, ExpoWidth-QtExpoWidth bits) ## WqGuarded(QtTotalWidth-2 downto 0) ## B(0, MantWidth-QtMantWidth bits)


  // * Approximate Multiplier Stage1
  val AxMultS1 = new AdderInt(Width=TotalWidth)
  AxMultS1.io.X := Wq_Extend_FP
  AxMultS1.io.Y := io.T_CIN_TC


  // * Guarding the result of A * Wq
  val GuardingAW = new GuardAW(TotalWidth=TotalWidth)
  GuardingAW.io.AW_In := AxMultS1.io.Sum
  GuardingAW.io.Wq_NotZero := SNC_Wq8.io.Wq_NotZero
  GuardingAW.io.A_Valid := io.A_Vld_CIN


  // * Approximated R = Wq * A
  io.R_FP := GuardingAW.io.AW_Out


  // * Cascade output
  io.T_COUT_TC := io.T_CIN_TC       // no pipeline
  io.A_Vld_COUT := io.A_Vld_CIN     // no pipeline
  io.Wq_COUT_FP := WqLockReg        // with pipeline

  // MARK: The Shared Partial Sum Adder is part of the PE, it is realized in the Tile level Component in our code.

}



object AxCore_SharedAdd_W8_PE_Gen extends App {
  // val ExpoWidth = 5
  // val MantWidth = 10
  // val ExpoWidth = 8
  // val MantWidth = 7
  val ExpoWidth = 8
  val MantWidth = 23

  Config.setGenSubDir(s"/PE/W8_E${ExpoWidth}M${MantWidth}")
  Config.spinal.generateVerilog(AxCore_SharedAdd_W8_PE(
    QtExpoWidth=4, QtMantWidth=3, ExpoWidth=ExpoWidth, MantWidth=MantWidth
  )).printRtl().mergeRTLSource()
}