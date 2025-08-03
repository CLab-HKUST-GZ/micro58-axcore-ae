/////////////////////////////////////////////////////////////
// Created by: Synopsys DC Ultra(TM) in wire load mode
// Version   : V-2023.12
// Date      : Sun Aug  3 17:39:40 2025
/////////////////////////////////////////////////////////////


module SNC_W4 ( Wq_FP_In, Wq_FmtSel, StochasticBit, Wq_NotZero, Wq_FP_Out );
  input [3:0] Wq_FP_In;
  input [1:0] Wq_FmtSel;
  output [5:0] Wq_FP_Out;
  input StochasticBit;
  output Wq_NotZero;
  wire   \Wq_FP_In[3] , n1, n2, n3, n4, n5, n6, n7, n8, n9, n10;
  assign Wq_FP_Out[5] = \Wq_FP_In[3] ;
  assign \Wq_FP_In[3]  = Wq_FP_In[3];

  INVD0BWP30P140 U3 ( .I(Wq_FP_In[0]), .ZN(n9) );
  INVD0BWP30P140 U4 ( .I(Wq_FmtSel[0]), .ZN(n7) );
  ND3D1BWP30P140 U5 ( .A1(Wq_FmtSel[1]), .A2(Wq_FP_In[2]), .A3(n7), .ZN(n2) );
  INVD0BWP30P140 U6 ( .I(Wq_FP_In[2]), .ZN(n5) );
  NR2D1BWP30P140 U7 ( .A1(n5), .A2(n7), .ZN(n6) );
  OAI211D0BWP30P140 U8 ( .A1(Wq_FP_In[0]), .A2(n6), .B(Wq_FP_In[1]), .C(
        Wq_FmtSel[1]), .ZN(n1) );
  OAI21D1BWP30P140 U9 ( .A1(n9), .A2(n2), .B(n1), .ZN(Wq_FP_Out[1]) );
  INVD0BWP30P140 U10 ( .I(Wq_FP_In[1]), .ZN(n4) );
  OAI21D1BWP30P140 U11 ( .A1(Wq_FmtSel[1]), .A2(n4), .B(n2), .ZN(Wq_FP_Out[3])
         );
  NR2D1BWP30P140 U12 ( .A1(Wq_FmtSel[1]), .A2(n5), .ZN(Wq_FP_Out[4]) );
  INVD0BWP30P140 U13 ( .I(Wq_FmtSel[1]), .ZN(n8) );
  OAI31D0BWP30P140 U14 ( .A1(StochasticBit), .A2(n7), .A3(n8), .B(Wq_FP_In[0]), 
        .ZN(n3) );
  ND3D1BWP30P140 U15 ( .A1(n5), .A2(n4), .A3(n3), .ZN(Wq_NotZero) );
  INR3D0BWP30P140 U16 ( .A1(n6), .B1(n9), .B2(n8), .ZN(Wq_FP_Out[0]) );
  AOI21D1BWP30P140 U17 ( .A1(Wq_FP_In[1]), .A2(n7), .B(n6), .ZN(n10) );
  AOI22D1BWP30P140 U18 ( .A1(Wq_FmtSel[1]), .A2(n10), .B1(n9), .B2(n8), .ZN(
        Wq_FP_Out[2]) );
endmodule


module AdderInt ( X, Y, Sum );
  input [15:0] X;
  input [15:0] Y;
  output [15:0] Sum;
  wire   \Y[7] , \Y[6] , \Y[5] , \Y[4] , \Y[3] , \Y[2] , \Y[1] , \Y[0] ,
         \intadd_0/CI , \intadd_0/n4 , \intadd_0/n3 , \intadd_0/n2 ,
         \intadd_0/n1 , n11, n12;
  assign Sum[7] = \Y[7] ;
  assign \Y[7]  = Y[7];
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

  FA1D0BWP30P140 \intadd_0/U5  ( .A(X[9]), .B(Y[9]), .CI(\intadd_0/CI ), .CO(
        \intadd_0/n4 ), .S(Sum[9]) );
  FA1D0BWP30P140 \intadd_0/U4  ( .A(X[10]), .B(Y[10]), .CI(\intadd_0/n4 ), 
        .CO(\intadd_0/n3 ), .S(Sum[10]) );
  FA1D0BWP30P140 \intadd_0/U3  ( .A(X[11]), .B(Y[11]), .CI(\intadd_0/n3 ), 
        .CO(\intadd_0/n2 ), .S(Sum[11]) );
  FA1D0BWP30P140 \intadd_0/U2  ( .A(X[12]), .B(Y[12]), .CI(\intadd_0/n2 ), 
        .CO(\intadd_0/n1 ), .S(Sum[12]) );
  AN2D0BWP30P140 U1 ( .A1(Y[8]), .A2(X[8]), .Z(\intadd_0/CI ) );
  IAO21D1BWP30P140 U2 ( .A1(Y[8]), .A2(X[8]), .B(\intadd_0/CI ), .ZN(Sum[8])
         );
  AN2D0BWP30P140 U3 ( .A1(\intadd_0/n1 ), .A2(Y[13]), .Z(n11) );
  IAO21D1BWP30P140 U4 ( .A1(\intadd_0/n1 ), .A2(Y[13]), .B(n11), .ZN(Sum[13])
         );
  ND3D1BWP30P140 U5 ( .A1(\intadd_0/n1 ), .A2(Y[13]), .A3(Y[14]), .ZN(n12) );
  OA21D0BWP30P140 U6 ( .A1(n11), .A2(Y[14]), .B(n12), .Z(Sum[14]) );
  XNR3UD0BWP30P140 U7 ( .A1(X[15]), .A2(Y[15]), .A3(n12), .ZN(Sum[15]) );
endmodule


module GuardAW ( AW_In, Wq_NotZero, A_Valid, AW_Out );
  input [15:0] AW_In;
  output [15:0] AW_Out;
  input Wq_NotZero, A_Valid;
  wire   n1;

  ND2D1BWP30P140 U2 ( .A1(Wq_NotZero), .A2(A_Valid), .ZN(n1) );
  INR2D1BWP30P140 U3 ( .A1(AW_In[0]), .B1(n1), .ZN(AW_Out[0]) );
  INR2D1BWP30P140 U4 ( .A1(AW_In[7]), .B1(n1), .ZN(AW_Out[7]) );
  INR2D1BWP30P140 U5 ( .A1(AW_In[6]), .B1(n1), .ZN(AW_Out[6]) );
  INR2D1BWP30P140 U6 ( .A1(AW_In[4]), .B1(n1), .ZN(AW_Out[4]) );
  INR2D1BWP30P140 U7 ( .A1(AW_In[5]), .B1(n1), .ZN(AW_Out[5]) );
  INR2D1BWP30P140 U8 ( .A1(AW_In[1]), .B1(n1), .ZN(AW_Out[1]) );
  INR2D1BWP30P140 U9 ( .A1(AW_In[3]), .B1(n1), .ZN(AW_Out[3]) );
  INR2D1BWP30P140 U10 ( .A1(AW_In[2]), .B1(n1), .ZN(AW_Out[2]) );
  INR2D1BWP30P140 U11 ( .A1(AW_In[8]), .B1(n1), .ZN(AW_Out[8]) );
  INR2D1BWP30P140 U12 ( .A1(AW_In[9]), .B1(n1), .ZN(AW_Out[9]) );
  INR2D1BWP30P140 U13 ( .A1(AW_In[10]), .B1(n1), .ZN(AW_Out[10]) );
  INR2D1BWP30P140 U14 ( .A1(AW_In[11]), .B1(n1), .ZN(AW_Out[11]) );
  INR2D1BWP30P140 U15 ( .A1(AW_In[12]), .B1(n1), .ZN(AW_Out[12]) );
  INR2D1BWP30P140 U16 ( .A1(AW_In[13]), .B1(n1), .ZN(AW_Out[13]) );
  INR2D1BWP30P140 U17 ( .A1(AW_In[14]), .B1(n1), .ZN(AW_Out[14]) );
  INR2D1BWP30P140 U18 ( .A1(AW_In[15]), .B1(n1), .ZN(AW_Out[15]) );
endmodule


module AxCore_SharedAdd_MPWq4_PE ( Wq_CIN_FP, Wq_COUT_FP, Wq_FmtSel, T_CIN_TC, 
        T_COUT_TC, A_Vld_CIN, A_Vld_COUT, R_FP, WqLock, clk, resetn );
  input [3:0] Wq_CIN_FP;
  output [3:0] Wq_COUT_FP;
  input [1:0] Wq_FmtSel;
  input [15:0] T_CIN_TC;
  output [15:0] T_COUT_TC;
  output [15:0] R_FP;
  input A_Vld_CIN, WqLock, clk, resetn;
  output A_Vld_COUT;
  wire   A_Vld_CIN, SNC_MPWq4_Wq_NotZero, n1, n2, n3, n4, net366, net367,
         net368, net369, net370, net371, net372, net373, net374, net375;
  wire   [5:0] SNC_MPWq4_Wq_FP_Out;
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

  SNC_W4 SNC_MPWq4 ( .Wq_FP_In(Wq_COUT_FP), .Wq_FmtSel(Wq_FmtSel), 
        .StochasticBit(T_CIN_TC[9]), .Wq_NotZero(SNC_MPWq4_Wq_NotZero), 
        .Wq_FP_Out(SNC_MPWq4_Wq_FP_Out) );
  AdderInt AxMultS1 ( .X({SNC_MPWq4_Wq_FP_Out[5], net366, net367, 
        SNC_MPWq4_Wq_FP_Out[4:0], net368, net369, net370, net371, net372, 
        net373, net374, net375}), .Y(T_CIN_TC), .Sum(AxMultS1_Sum) );
  GuardAW GuardingAW ( .AW_In(AxMultS1_Sum), .Wq_NotZero(SNC_MPWq4_Wq_NotZero), 
        .A_Valid(A_Vld_CIN), .AW_Out(R_FP) );
  DFCNQD1BWP30P140 \WqLockReg_reg[3]  ( .D(n4), .CP(clk), .CDN(resetn), .Q(
        Wq_COUT_FP[3]) );
  DFCNQD1BWP30P140 \WqLockReg_reg[1]  ( .D(n2), .CP(clk), .CDN(resetn), .Q(
        Wq_COUT_FP[1]) );
  DFCNQD1BWP30P140 \WqLockReg_reg[0]  ( .D(n1), .CP(clk), .CDN(resetn), .Q(
        Wq_COUT_FP[0]) );
  DFCNQD1BWP30P140 \WqLockReg_reg[2]  ( .D(n3), .CP(clk), .CDN(resetn), .Q(
        Wq_COUT_FP[2]) );
  CKMUX2D0BWP30P140 U8 ( .I0(Wq_CIN_FP[3]), .I1(Wq_COUT_FP[3]), .S(WqLock), 
        .Z(n4) );
  CKMUX2D0BWP30P140 U9 ( .I0(Wq_CIN_FP[2]), .I1(Wq_COUT_FP[2]), .S(WqLock), 
        .Z(n3) );
  CKMUX2D0BWP30P140 U10 ( .I0(Wq_CIN_FP[0]), .I1(Wq_COUT_FP[0]), .S(WqLock), 
        .Z(n1) );
  CKMUX2D0BWP30P140 U11 ( .I0(Wq_CIN_FP[1]), .I1(Wq_COUT_FP[1]), .S(WqLock), 
        .Z(n2) );
endmodule

