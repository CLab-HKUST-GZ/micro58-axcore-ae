package AxCore.Basics

import spinal.core._
import spinal.core.sim._
import AxCore.Config
import scala.language.postfixOps
import AxCore.Operators.AdderInt


case class AxDequant(ExpoWidth: Int, MantWidth: Int) extends Component {

  val TotalWidth = 1 + ExpoWidth + MantWidth

  val io = new Bundle {
    val PSum_FP               = in  Bits(TotalWidth bits)
    val S_minus_B2_plus_C2_TC = in  Bits(TotalWidth bits)
    val Oact_FP               = out Bits(TotalWidth bits)
  }
  noIoPrefix()


  // * Approximate Multiplier Stage2
  val AxMultS2 = AdderInt(Width=TotalWidth)
  AxMultS2.io.X := io.PSum_FP
  AxMultS2.io.Y := io.S_minus_B2_plus_C2_TC

  io.Oact_FP := AxMultS2.io.Sum

}



object AxDequantGen extends App {
  Config.setGenSubDir("/AxDequant")
  Config.spinal.generateVerilog(AxDequant(ExpoWidth=5, MantWidth=10)).printRtl().mergeRTLSource()
}