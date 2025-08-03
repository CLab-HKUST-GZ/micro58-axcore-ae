package AxCore.Testing.GoldenModels

import scala.language.postfixOps


case class SncOutput(Wq_FP_Out: BigInt, Wq_NotZero: Boolean)

object GoldenModel_SNC_W4 {
  def calculate(Wq_FP_In: BigInt, Wq_FmtSel: BigInt, StochasticBit: Boolean): SncOutput = {

    val Sign = (Wq_FP_In >> 3) & 1
    val Content = Wq_FP_In & 0x7       // Get the low 3 bits
    var Wq_FP_Out: BigInt = 0
    var Wq_NotZero: Boolean = false

    // Deciding output according to Wq_FmtSel and Wq_FP_In
    if (Wq_FmtSel == 0 || Wq_FmtSel == 1) {    // Wq_FmtSel = 00,01 for E3M0
      Wq_FP_Out = (Sign << 5) | (Content << 2)      // Sign ## Content ## "00"
      Wq_NotZero = (Content != 0)
    } else if (Wq_FmtSel == 2) {               // Wq_FmtSel = 10 for E2M1
      if (Content <= 1) {
        Wq_FP_Out = Sign << 5                       // Sign ## "000_00"
      } else {
        Wq_FP_Out = (Sign << 5) | (Content << 1)    // Sign ## "0" ## Content ## "0"
      }
      Wq_NotZero = (Content != 0)
    } else {                                   // Wq_FmtSel = 11 for E1M2
      if (Content <= 2) {
        Wq_FP_Out = Sign << 5                       // Sign ## "000_00"
      } else if (Content == 3) {
        Wq_FP_Out = (Sign << 5) | 2                 // Sign ## "00" ## "010"
      } else {
        Wq_FP_Out = (Sign << 5) | Content           // Sign ## "00" ## Content
      }
      val needRandomize = (Content == 1)
      if (needRandomize) {
        Wq_NotZero = StochasticBit
      } else {
        Wq_NotZero = (Content != 0)
      }
    }

    // Return outputs
    SncOutput(Wq_FP_Out = Wq_FP_Out, Wq_NotZero = Wq_NotZero)
  }
}