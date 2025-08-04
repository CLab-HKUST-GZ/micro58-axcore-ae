package AxCore.Basics

import spinal.core._
import spinal.core.sim._
import AxCore.Config
import scala.language.postfixOps


case class GuardAW(TotalWidth: Int) extends Component {
  val io = new Bundle {
    val AW_In      = in  Bits(TotalWidth bits)  simPublic()
    val Wq_NotZero = in  Bool()
    val A_Valid    = in  Bool()

    val AW_Out     = out Bits(TotalWidth bits)
  }
  noIoPrefix()

  // Check if AW_In (view as 2's complement) >= 0
  val SignTC = io.AW_In(TotalWidth-1)

  val Valid = io.Wq_NotZero & io.A_Valid

  io.AW_Out := Mux(Valid, io.AW_In, B(0))

}
