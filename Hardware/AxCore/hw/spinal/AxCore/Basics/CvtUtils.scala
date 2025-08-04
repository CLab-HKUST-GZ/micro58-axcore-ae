package AxCore.Basics

import scala.math._
import scala.math.pow

class CvtUtils {}


object FP2BinCvt {

  def FloatToFPAnyBin(
                       f          : Double,                  // Floating-Point format
                       ExpoWidth  : Int,
                       MantWidth  : Int,
                       CustomBias : Option[Int] = None,
                       withNaNInf : Boolean = true
                     ): String = {                           // Return FP's binary format

    def toBinary(n: Int, width: Int): String = {
      if (width <= 0) ""
      else {
        val binStr = n.toBinaryString
        val padLength = width - binStr.length
        val padded = if (padLength > 0) "0" * padLength + binStr else binStr
        padded.takeRight(width)
      }
    }

    val sign = if (f >= 0) 0 else 1

    if (f == 0.0)
      return s"${sign}_${"0" * ExpoWidth}_${"0" * MantWidth}"

    if (withNaNInf) {
      if (f.isNaN)
        return s"0_${"1" * ExpoWidth}_${"0" * (MantWidth - 1)}1"

      if (f.isInfinite)
        return s"${if (f > 0) 0 else 1}_${"1" * ExpoWidth}_${"0" * MantWidth}"
    } else if (f.isNaN || f.isInfinite) {
      return s"${sign}_${"1" * ExpoWidth}_${"1" * MantWidth}"
    }

    val absF = abs(f)
    val expo = floor(log(absF) / log(2)).toInt
    var mant = absF / pow(2, expo) - 1.0

    val fpBias = CustomBias.getOrElse((1 << (ExpoWidth - 1)) - 1)

    val (maxExponent, exponentMax) = if (withNaNInf)
      ((1 << ExpoWidth) - 2, (1 << ExpoWidth) - 2 - fpBias)
    else
      ((1 << ExpoWidth) - 1, (1 << ExpoWidth) - 1 - fpBias)

    val maxNormal = pow(2, exponentMax) * (2 - pow(2, -MantWidth))
    val minNormal = pow(2, 1 - fpBias)

    val (expoVal, mantVal) =
      if (absF < minNormal) {
        val scaledMant = absF / pow(2, 1 - fpBias)
        val mantBits = round(scaledMant * (1 << MantWidth)).min((1 << MantWidth) - 1)
        (0, mantBits)
      } else if (absF > maxNormal) {
        return s"${sign}_${"1" * ExpoWidth}_${"1" * MantWidth}"
      } else {
        val adjustedExpo = expo + fpBias
        val mantBits = round(mant * (1 << MantWidth)).min((1 << MantWidth) - 1)
        (adjustedExpo, mantBits)
      }

    s"${sign}_${toBinary(expoVal, ExpoWidth)}_${toBinary(mantVal.toInt, MantWidth)}"
  }

}



object Bin2FPCvt {

  def FPAnyBinToFloat(
                       FPBin      : String,                  // FP's binary format
                       ExpoWidth  : Int,
                       MantWidth  : Int,
                       CustomBias : Option[Int] = None,
                       WithNaNInf : Boolean = true
                     ): Double = {                           // Return Floating-Point format

    val TotalWidth = 1 + ExpoWidth + MantWidth
    val cleaned = FPBin.replace("_", "")
    require(cleaned.length == TotalWidth, s"Input must be a $TotalWidth-bit binary string")

    val FPBias = CustomBias.getOrElse((1 << (ExpoWidth - 1)) - 1)

    // * Extract each part
    val Sign = Integer.parseInt(cleaned.substring(0, 1), 2)
    val Expo = Integer.parseInt(cleaned.substring(1, 1 + ExpoWidth), 2)
    val Mant = Integer.parseInt(cleaned.substring(1 + ExpoWidth), 2)

    val MaxExpo = (1 << ExpoWidth) - 1

    // * Sub-Normal Value
    if (Expo == 0) {
      if (Mant == 0) 0.0
      else {
        val SignFactor = if (Sign == 0) 1.0 else -1.0
        SignFactor * pow(2, 1 - FPBias) * (Mant.toDouble / pow(2, MantWidth))
      }

    // * NaN of Inf
    } else if (Expo == MaxExpo && WithNaNInf) {
      if (Mant == 0) {
        if (Sign == 0) Double.PositiveInfinity else Double.NegativeInfinity
      } else {
        Double.NaN
      }

    // * Normal Value
    } else {
      val TrueExpo = Expo - FPBias
      val TrueMant = 1.0 + Mant.toDouble / pow(2, MantWidth)
      val SignFactor = if (Sign == 0) 1.0 else -1.0
      SignFactor * pow(2, TrueExpo) * TrueMant
    }
  }

}



