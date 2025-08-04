package AxCore.Operators

import spinal.core._
import spinal.core.sim._
import AxCore.Config
import scala.language.postfixOps


case class AdderInt(Width: Int) extends Component {
  val io = new Bundle {
    val X   = in  Bits(Width bits)
    val Y   = in  Bits(Width bits)
    val Sum = out Bits(Width bits)
  }
  noIoPrefix()

  io.Sum := (io.X.asUInt + io.Y.asUInt).setName(s"SumSameWidth").asBits

}



object AdderInt_Gen extends App {
  Config.setGenSubDir("/AdderInt")
  Config.spinal.generateVerilog(AdderInt(Width=16)).printRtl().mergeRTLSource()
}