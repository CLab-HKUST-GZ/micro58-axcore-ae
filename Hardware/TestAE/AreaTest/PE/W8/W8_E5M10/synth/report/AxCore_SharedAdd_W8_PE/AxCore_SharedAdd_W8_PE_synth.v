/////////////////////////////////////////////////////////////
// Created by: Synopsys DC Ultra(TM) in wire load mode
// Version   : V-2023.12
// Date      : Sun Aug  3 16:38:07 2025
/////////////////////////////////////////////////////////////


module SNC_W8 ( Wq_FP_In, Wq_FmtSel, StochasticBit, Wq_NotZero, Wq_FP_Out );
  input [7:0] Wq_FP_In;
  input [1:0] Wq_FmtSel;
  output [7:0] Wq_FP_Out;
  input StochasticBit;
  output Wq_NotZero;
  wire   \Wq_FP_In[7] , \Wq_FP_In[6] , \Wq_FP_In[5] , \Wq_FP_In[4] ,
         \Wq_FP_In[3] , n1, n2, n3, n4, n5, n6;
  assign Wq_FP_Out[7] = \Wq_FP_In[7] ;
  assign \Wq_FP_In[7]  = Wq_FP_In[7];
  assign Wq_FP_Out[6] = \Wq_FP_In[6] ;
  assign \Wq_FP_In[6]  = Wq_FP_In[6];
  assign Wq_FP_Out[5] = \Wq_FP_In[5] ;
  assign \Wq_FP_In[5]  = Wq_FP_In[5];
  assign Wq_FP_Out[4] = \Wq_FP_In[4] ;
  assign \Wq_FP_In[4]  = Wq_FP_In[4];
  assign Wq_FP_Out[3] = \Wq_FP_In[3] ;
  assign \Wq_FP_In[3]  = Wq_FP_In[3];

  OAI32D0BWP30P140 U3 ( .A1(n3), .A2(n5), .A3(n2), .B1(n6), .B2(n1), .ZN(
        Wq_FP_Out[1]) );
  NR4D0BWP30P140 U4 ( .A1(\Wq_FP_In[6] ), .A2(\Wq_FP_In[5] ), .A3(
        \Wq_FP_In[4] ), .A4(\Wq_FP_In[3] ), .ZN(n6) );
  INVD0BWP30P140 U5 ( .I(Wq_FP_In[0]), .ZN(n2) );
  NR2D1BWP30P140 U6 ( .A1(n6), .A2(n2), .ZN(Wq_FP_Out[0]) );
  INVD0BWP30P140 U7 ( .I(Wq_FP_In[1]), .ZN(n1) );
  INVD0BWP30P140 U8 ( .I(Wq_FP_In[2]), .ZN(n5) );
  AOI21D1BWP30P140 U9 ( .A1(n6), .A2(n1), .B(n5), .ZN(Wq_FP_Out[2]) );
  INVD0BWP30P140 U10 ( .I(n6), .ZN(n3) );
  OAI21D1BWP30P140 U11 ( .A1(Wq_FP_In[0]), .A2(StochasticBit), .B(Wq_FP_In[1]), 
        .ZN(n4) );
  ND3D1BWP30P140 U12 ( .A1(n6), .A2(n5), .A3(n4), .ZN(Wq_NotZero) );
endmodule


module AdderInt ( X, Y, Sum );
  input [15:0] X;
  input [15:0] Y;
  output [15:0] Sum;
  wire   \Y[6] , \Y[5] , \Y[4] , \Y[3] , \Y[2] , \Y[1] , \Y[0] , \intadd_0/CI ,
         \intadd_0/n6 , \intadd_0/n5 , \intadd_0/n4 , \intadd_0/n3 ,
         \intadd_0/n2 , \intadd_0/n1 , n9;
  assign Sum[6] = \Y[6] ;
  assign \Y[6]  = Y[6];
  assign Sum[5] = \Y[5] ;
  assign \Y[5]  = Y[5];
  assign Sum[4] = \Y[4] ;
  assign \Y[4]  = Y[4];
  assign Sum[3] = \Y[3] ;
  assign \Y[3]  = Y[3];
  assign Sum[2] = \Y[2] ;
  assign \Y[2]  = Y[2];
  assign Sum[1] = \Y[1] ;
  assign \Y[1]  = Y[1];
  assign Sum[0] = \Y[0] ;
  assign \Y[0]  = Y[0];

  FA1D0BWP30P140 \intadd_0/U7  ( .A(X[8]), .B(Y[8]), .CI(\intadd_0/CI ), .CO(
        \intadd_0/n6 ), .S(Sum[8]) );
  FA1D0BWP30P140 \intadd_0/U6  ( .A(X[9]), .B(Y[9]), .CI(\intadd_0/n6 ), .CO(
        \intadd_0/n5 ), .S(Sum[9]) );
  FA1D0BWP30P140 \intadd_0/U5  ( .A(X[10]), .B(Y[10]), .CI(\intadd_0/n5 ), 
        .CO(\intadd_0/n4 ), .S(Sum[10]) );
  FA1D0BWP30P140 \intadd_0/U4  ( .A(X[11]), .B(Y[11]), .CI(\intadd_0/n4 ), 
        .CO(\intadd_0/n3 ), .S(Sum[11]) );
  FA1D0BWP30P140 \intadd_0/U3  ( .A(X[12]), .B(Y[12]), .CI(\intadd_0/n3 ), 
        .CO(\intadd_0/n2 ), .S(Sum[12]) );
  FA1D0BWP30P140 \intadd_0/U2  ( .A(X[13]), .B(Y[13]), .CI(\intadd_0/n2 ), 
        .CO(\intadd_0/n1 ), .S(Sum[13]) );
  XNR3UD0BWP30P140 U1 ( .A1(X[15]), .A2(Y[15]), .A3(n9), .ZN(Sum[15]) );
  AN2D0BWP30P140 U2 ( .A1(Y[7]), .A2(X[7]), .Z(\intadd_0/CI ) );
  IAO21D1BWP30P140 U3 ( .A1(Y[7]), .A2(X[7]), .B(\intadd_0/CI ), .ZN(Sum[7])
         );
  ND2D1BWP30P140 U4 ( .A1(\intadd_0/n1 ), .A2(Y[14]), .ZN(n9) );
  OA21D0BWP30P140 U5 ( .A1(\intadd_0/n1 ), .A2(Y[14]), .B(n9), .Z(Sum[14]) );
endmodule


module GuardAW ( AW_In, Wq_NotZero, A_Valid, AW_Out );
  input [15:0] AW_In;
  output [15:0] AW_Out;
  input Wq_NotZero, A_Valid;
  wire   n1;

  ND2D1BWP30P140 U2 ( .A1(Wq_NotZero), .A2(A_Valid), .ZN(n1) );
  INR2D1BWP30P140 U3 ( .A1(AW_In[0]), .B1(n1), .ZN(AW_Out[0]) );
  INR2D1BWP30P140 U4 ( .A1(AW_In[6]), .B1(n1), .ZN(AW_Out[6]) );
  INR2D1BWP30P140 U5 ( .A1(AW_In[5]), .B1(n1), .ZN(AW_Out[5]) );
  INR2D1BWP30P140 U6 ( .A1(AW_In[4]), .B1(n1), .ZN(AW_Out[4]) );
  INR2D1BWP30P140 U7 ( .A1(AW_In[3]), .B1(n1), .ZN(AW_Out[3]) );
  INR2D1BWP30P140 U8 ( .A1(AW_In[1]), .B1(n1), .ZN(AW_Out[1]) );
  INR2D1BWP30P140 U9 ( .A1(AW_In[2]), .B1(n1), .ZN(AW_Out[2]) );
  INR2D1BWP30P140 U10 ( .A1(AW_In[7]), .B1(n1), .ZN(AW_Out[7]) );
  INR2D1BWP30P140 U11 ( .A1(AW_In[8]), .B1(n1), .ZN(AW_Out[8]) );
  INR2D1BWP30P140 U12 ( .A1(AW_In[9]), .B1(n1), .ZN(AW_Out[9]) );
  INR2D1BWP30P140 U13 ( .A1(AW_In[10]), .B1(n1), .ZN(AW_Out[10]) );
  INR2D1BWP30P140 U14 ( .A1(AW_In[11]), .B1(n1), .ZN(AW_Out[11]) );
  INR2D1BWP30P140 U15 ( .A1(AW_In[12]), .B1(n1), .ZN(AW_Out[12]) );
  INR2D1BWP30P140 U16 ( .A1(AW_In[13]), .B1(n1), .ZN(AW_Out[13]) );
  INR2D1BWP30P140 U17 ( .A1(AW_In[14]), .B1(n1), .ZN(AW_Out[14]) );
  INR2D1BWP30P140 U18 ( .A1(AW_In[15]), .B1(n1), .ZN(AW_Out[15]) );
endmodule


module AxCore_SharedAdd_W8_PE ( Wq_CIN_FP, Wq_COUT_FP, T_CIN_TC, T_COUT_TC, 
        A_Vld_CIN, A_Vld_COUT, R_FP, WqLock, clk, resetn );
  input [7:0] Wq_CIN_FP;
  output [7:0] Wq_COUT_FP;
  input [15:0] T_CIN_TC;
  output [15:0] T_COUT_TC;
  output [15:0] R_FP;
  input A_Vld_CIN, WqLock, clk, resetn;
  output A_Vld_COUT;
  wire   A_Vld_CIN, SNC_Wq8_Wq_NotZero, n1, n2, n3, n4, n5, n6, n7, n8, net294,
         net295, net296, net297, net298, net299, net300, net301, net302,
         net303;
  wire   [7:0] SNC_Wq8_Wq_FP_Out;
  wire   [15:0] AxMultS1_Sum;
  assign T_COUT_TC[15] = T_CIN_TC[15];
  assign T_COUT_TC[14] = T_CIN_TC[14];
  assign T_COUT_TC[13] = T_CIN_TC[13];
  assign T_COUT_TC[12] = T_CIN_TC[12];
  assign T_COUT_TC[11] = T_CIN_TC[11];
  assign T_COUT_TC[10] = T_CIN_TC[10];
  assign T_COUT_TC[9] = T_CIN_TC[9];
  assign T_COUT_TC[8] = T_CIN_TC[8];
  assign T_COUT_TC[7] = T_CIN_TC[7];
  assign T_COUT_TC[6] = T_CIN_TC[6];
  assign T_COUT_TC[5] = T_CIN_TC[5];
  assign T_COUT_TC[4] = T_CIN_TC[4];
  assign T_COUT_TC[3] = T_CIN_TC[3];
  assign T_COUT_TC[2] = T_CIN_TC[2];
  assign T_COUT_TC[1] = T_CIN_TC[1];
  assign T_COUT_TC[0] = T_CIN_TC[0];
  assign A_Vld_COUT = A_Vld_CIN;

  SNC_W8 SNC_Wq8 ( .Wq_FP_In(Wq_COUT_FP), .Wq_FmtSel({net302, net303}), 
        .StochasticBit(T_CIN_TC[9]), .Wq_NotZero(SNC_Wq8_Wq_NotZero), 
        .Wq_FP_Out(SNC_Wq8_Wq_FP_Out) );
  AdderInt AxMultS1 ( .X({SNC_Wq8_Wq_FP_Out[7], net294, SNC_Wq8_Wq_FP_Out[6:0], 
        net295, net296, net297, net298, net299, net300, net301}), .Y(T_CIN_TC), 
        .Sum(AxMultS1_Sum) );
  GuardAW GuardingAW ( .AW_In(AxMultS1_Sum), .Wq_NotZero(SNC_Wq8_Wq_NotZero), 
        .A_Valid(A_Vld_CIN), .AW_Out(R_FP) );
  DFCNQD1BWP30P140 \WqLockReg_reg[7]  ( .D(n8), .CP(clk), .CDN(resetn), .Q(
        Wq_COUT_FP[7]) );
  DFCNQD1BWP30P140 \WqLockReg_reg[0]  ( .D(n1), .CP(clk), .CDN(resetn), .Q(
        Wq_COUT_FP[0]) );
  DFCNQD1BWP30P140 \WqLockReg_reg[2]  ( .D(n3), .CP(clk), .CDN(resetn), .Q(
        Wq_COUT_FP[2]) );
  DFCNQD1BWP30P140 \WqLockReg_reg[1]  ( .D(n2), .CP(clk), .CDN(resetn), .Q(
        Wq_COUT_FP[1]) );
  DFCNQD1BWP30P140 \WqLockReg_reg[6]  ( .D(n7), .CP(clk), .CDN(resetn), .Q(
        Wq_COUT_FP[6]) );
  DFCNQD1BWP30P140 \WqLockReg_reg[5]  ( .D(n6), .CP(clk), .CDN(resetn), .Q(
        Wq_COUT_FP[5]) );
  DFCNQD1BWP30P140 \WqLockReg_reg[4]  ( .D(n5), .CP(clk), .CDN(resetn), .Q(
        Wq_COUT_FP[4]) );
  DFCNQD1BWP30P140 \WqLockReg_reg[3]  ( .D(n4), .CP(clk), .CDN(resetn), .Q(
        Wq_COUT_FP[3]) );
  CKMUX2D0BWP30P140 U12 ( .I0(Wq_CIN_FP[2]), .I1(Wq_COUT_FP[2]), .S(WqLock), 
        .Z(n3) );
  CKMUX2D0BWP30P140 U13 ( .I0(Wq_CIN_FP[7]), .I1(Wq_COUT_FP[7]), .S(WqLock), 
        .Z(n8) );
  CKMUX2D0BWP30P140 U14 ( .I0(Wq_CIN_FP[0]), .I1(Wq_COUT_FP[0]), .S(WqLock), 
        .Z(n1) );
  CKMUX2D0BWP30P140 U15 ( .I0(Wq_CIN_FP[1]), .I1(Wq_COUT_FP[1]), .S(WqLock), 
        .Z(n2) );
  CKMUX2D0BWP30P140 U16 ( .I0(Wq_CIN_FP[3]), .I1(Wq_COUT_FP[3]), .S(WqLock), 
        .Z(n4) );
  CKMUX2D0BWP30P140 U17 ( .I0(Wq_CIN_FP[4]), .I1(Wq_COUT_FP[4]), .S(WqLock), 
        .Z(n5) );
  CKMUX2D0BWP30P140 U18 ( .I0(Wq_CIN_FP[5]), .I1(Wq_COUT_FP[5]), .S(WqLock), 
        .Z(n6) );
  CKMUX2D0BWP30P140 U19 ( .I0(Wq_CIN_FP[6]), .I1(Wq_COUT_FP[6]), .S(WqLock), 
        .Z(n7) );
endmodule

