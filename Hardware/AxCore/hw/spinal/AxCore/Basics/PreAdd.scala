package AxCore.Basics

import spinal.core._
import spinal.core.sim._
import AxCore.Config
import scala.language.postfixOps
import AxCore.Operators.AdderInt
import scala.math.pow


// MARK: PreAdd Unit for A - B1 + C1
case class PreAdd(QtTotalWidth : Int, ExpoWidth: Int, MantWidth: Int) extends Component {

  val TotalWidth = 1 + ExpoWidth + MantWidth

  val io = new Bundle {
    val A_FP      = in  Bits(TotalWidth bits)
    val Wq_FmtSel = in  Bits(2 bits)                 // Wq Format Select. 00,01 for E3M0, 10 for E2M1, 11 for E1M2

    val T_TC      = out Bits(TotalWidth bits)        // T in Two's Complement
    val A_Valid   = out Bool()
  }
  noIoPrefix()


  // * Compensation Stage 1 (k=0)
  val C1 = MantWidth match {
    case 10 => 43    // For A in E5M10, C1=43
    case 7  => 5     // For A in E8M7,  C1=5
    case _  => 0
  }


  // * -B1 = [2^(QtExpoWidth-1)-1] << MantWidth
  def QtNegB(QtExpoWidth: Int): Int = {
    -((pow(2, QtExpoWidth-1) - 1) * pow(2, MantWidth)).toInt
  }

  val C1_minus_B1 = Bits(TotalWidth bits)

  if (QtTotalWidth == 4) {    // * For W4
    switch(io.Wq_FmtSel) {
      is (B"00") { C1_minus_B1 := S(C1+QtNegB(3), TotalWidth bits).asBits }    // For Wq in E3M0
      is (B"01") { C1_minus_B1 := S(C1+QtNegB(3), TotalWidth bits).asBits }    // For Wq in E3M0
      is (B"10") { C1_minus_B1 := S(C1+QtNegB(2), TotalWidth bits).asBits }    // For Wq in E2M1
      is (B"11") { C1_minus_B1 := S(C1+QtNegB(1), TotalWidth bits).asBits }    // For Wq in E1M2
    }
  } else if (QtTotalWidth == 8) {    // * For W8
    C1_minus_B1 := S(C1+QtNegB(4), TotalWidth bits).asBits
  }


  // * A - B1 + C1
  val Adder1 = AdderInt(Width=TotalWidth)
  Adder1.io.X := io.A_FP
  Adder1.io.Y := C1_minus_B1

  io.T_TC := Adder1.io.Sum

  io.A_Valid := io.A_FP.orR    // Reduced OR

}



object PreAddGen extends App {
 val QtTotalWidth = 4
//   val QtTotalWidth = 8
  val ExpoWidth    = 5
  val MantWidth    = 10
  Config.setGenSubDir("/PreAdd")
  Config.spinal.generateVerilog(PreAdd(QtTotalWidth=QtTotalWidth, ExpoWidth=ExpoWidth, MantWidth=MantWidth)).printRtl().mergeRTLSource()
}