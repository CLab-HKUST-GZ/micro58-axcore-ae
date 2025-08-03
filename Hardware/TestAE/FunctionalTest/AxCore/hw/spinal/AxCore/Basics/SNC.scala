package AxCore.Basics

import spinal.core._
import spinal.core.sim._
import AxCore.Config
import scala.language.postfixOps


// SNC_W4 & SNC_W8 extend from this
class SNC_Base(QtTotalWidth: Int, OutputWidth: Int) extends Component {

  val io = new Bundle {
    val Wq_FP_In      = in  Bits(QtTotalWidth bits)
    val Wq_FmtSel     = in  Bits(2 bits)    // 00,01 for E3M0, 10 for E2M1, 11 for E1M2. Only valid in W4.
    val StochasticBit = in  Bool()                    simPublic()

    val Wq_NotZero    = out Bool()                    simPublic()
    val Wq_FP_Out     = out Bits(OutputWidth bits)    simPublic()
  }
  noIoPrefix()

}



// MARK: Adaptive-format in W4: for E3M0, E2M1, E1M2
case class SNC_W4(QtTotalWidth: Int=4, OutputWidth: Int=6) extends SNC_Base(QtTotalWidth, OutputWidth) {

  val Sign    = io.Wq_FP_In(3)
  val Content = io.Wq_FP_In(2 downto 0)
  val Decide  = io.Wq_FmtSel ## Content

  switch(Decide) {
    // * E2M1 (FmtSel = "10")
    is (B"10__00_0") { io.Wq_FP_Out := Sign ## B("0") ## B("00_0") ## B("0") }    // Subnormal
    is (B"10__00_1") { io.Wq_FP_Out := Sign ## B("0") ## B("00_0") ## B("0") }    // Subnormal
    is (B"10__01_0") { io.Wq_FP_Out := Sign ## B("0") ## Content   ## B("0") }
    is (B"10__01_1") { io.Wq_FP_Out := Sign ## B("0") ## Content   ## B("0") }
    is (B"10__10_0") { io.Wq_FP_Out := Sign ## B("0") ## Content   ## B("0") }
    is (B"10__10_1") { io.Wq_FP_Out := Sign ## B("0") ## Content   ## B("0") }
    is (B"10__11_0") { io.Wq_FP_Out := Sign ## B("0") ## Content   ## B("0") }
    is (B"10__11_1") { io.Wq_FP_Out := Sign ## B("0") ## Content   ## B("0") }

    // * E1M2 (FmtSel = "11")
    is (B"11__0_00") { io.Wq_FP_Out := Sign ## B("00") ## B("0_00") }    // Subnormal
    is (B"11__0_01") { io.Wq_FP_Out := Sign ## B("00") ## B("0_00") }    // Subnormal
    is (B"11__0_10") { io.Wq_FP_Out := Sign ## B("00") ## B("0_00") }    // Subnormal
    is (B"11__0_11") { io.Wq_FP_Out := Sign ## B("00") ## B("0_10") }    // Subnormal
    is (B"11__1_00") { io.Wq_FP_Out := Sign ## B("00") ## Content   }
    is (B"11__1_01") { io.Wq_FP_Out := Sign ## B("00") ## Content   }
    is (B"11__1_10") { io.Wq_FP_Out := Sign ## B("00") ## Content   }
    is (B"11__1_11") { io.Wq_FP_Out := Sign ## B("00") ## Content   }

    // * E3M0
    default { io.Wq_FP_Out := Sign ## Content ## B("00") }
  }

  // * Random selection & Zero Flag
  val NeedRandomize = (Decide === B"11__0_01")
  io.Wq_NotZero := Mux(NeedRandomize, io.StochasticBit, io.Wq_FP_In(2 downto 0).orR)

}



// MARK: For FP8 E4M3
case class SNC_W8(QtTotalWidth: Int=8, OutputWidth: Int=8) extends SNC_Base(QtTotalWidth, OutputWidth) {

  val WqIsNorm = io.Wq_FP_In(6 downto 3).orR    // E4 != 0
  val WqMant = io.Wq_FP_In(2 downto 0)          // Original Mantissa

  val SubNormCvt = Bits(3 bits)                 // Subnormal converted Mantissa
  switch(WqMant) {
    is (B"101") { SubNormCvt := B("010") }
    is (B"110") { SubNormCvt := B("100") }
    is (B"111") { SubNormCvt := B("110") }
    default     { SubNormCvt := B("000") }
  }

  when(WqIsNorm) {
    io.Wq_FP_Out := io.Wq_FP_In
  } otherwise {
    io.Wq_FP_Out := io.Wq_FP_In(7 downto 3) ## SubNormCvt
  }

  // * Random selection & Zero Flag
  val NeedRandomize = (io.Wq_FP_In(6 downto 0) === B"0000_010")
  io.Wq_NotZero := Mux(NeedRandomize, io.StochasticBit, WqIsNorm | io.Wq_FP_In(2 downto 1).orR)

}




object SNC_Gen extends App {
  Config.setGenSubDir("/SNC")
  Config.spinal.generateVerilog(SNC_W4()).printRtl().mergeRTLSource()
//   Config.spinal.generateVerilog(SNC_W8()).printRtl().mergeRTLSource()
}