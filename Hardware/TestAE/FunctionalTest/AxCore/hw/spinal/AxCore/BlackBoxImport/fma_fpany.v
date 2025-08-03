module fma_fpany
#(
    parameter E_MUL = 5,
    parameter M_MUL = 10,
    parameter E_ADD = 8,
    parameter M_ADD = 23
)
(

    input                       clk     ,

    input   [E_MUL + M_MUL  :0] src0    ,   //input at d0
    input   [E_MUL + M_MUL  :0] src1    ,   //input at d0
    input   [E_ADD + M_ADD  :0] src2    ,   //input at d0
    output  [E_ADD + M_ADD  :0] result  ,   //output at d1
    output                      overflow
);
////////////////////////////////////////////////
// unpack
wire                src0_sign;
wire [E_MUL - 1 :0] src0_exp;
wire [M_MUL - 1 :0] src0_man;
wire                src1_sign;
wire [E_MUL - 1 :0] src1_exp;
wire [M_MUL - 1 :0] src1_man;
wire                src2_sign;
wire [E_ADD - 1 :0] src2_exp;
wire [M_ADD - 1 :0] src2_man;

assign src0_sign = src0[E_MUL + M_MUL];
assign src0_exp  = src0[E_MUL + M_MUL - 1:M_MUL];
assign src0_man  = src0[M_MUL - 1:0];
assign src1_sign = src1[E_MUL + M_MUL];
assign src1_exp  = src1[E_MUL + M_MUL - 1:M_MUL];
assign src1_man  = src1[M_MUL - 1:0];
assign src2_sign = src2[E_ADD + M_ADD];
assign src2_exp  = src2[E_ADD + M_ADD - 1:M_ADD];
assign src2_man  = src2[M_ADD - 1:0];
////////////////////////////////////////////////
// whether the number is normal
wire                src0_normal;
wire                src1_normal;
wire                src2_normal;
wire [E_MUL - 1 :0] src0_exp_normal;
wire [E_MUL - 1 :0] src1_exp_normal;
wire [E_ADD - 1 :0] src2_exp_normal;
wire [M_MUL     :0] src0_man_normal;
wire [M_MUL     :0] src1_man_normal;
wire [M_ADD     :0] src2_man_normal;

assign src0_normal = src0_exp != 'b0;
assign src1_normal = src1_exp != 'b0;
assign src2_normal = src2_exp != 'b0;
assign src0_exp_normal = src0_exp + {{(E_MUL - 1){1'b0}}, ~src0_normal};
assign src1_exp_normal = src1_exp + {{(E_MUL - 1){1'b0}}, ~src1_normal};
assign src2_exp_normal = src2_exp + {{(E_ADD - 1){1'b0}}, ~src2_normal};
assign src0_man_normal = {src0_normal, src0_man};
assign src1_man_normal = {src1_normal, src1_man};
assign src2_man_normal = {src2_normal, src2_man};
////////////////////////////////////////////////
localparam [E_ADD:0] BIAS_MUL = (1 << (E_MUL - 1)) - 1;
localparam [E_ADD:0] BIAS_ADD = (1 << (E_ADD - 1)) - 1;
localparam [E_ADD:0] BIAS_NEW = BIAS_ADD - 2*BIAS_MUL;
////////////////////////////////////////////////
// multiplication
wire                        mul_sign;
wire [E_ADD             :0] mul_exp;
wire [2*(M_MUL + 1) - 1 :0] mul_man; // xx.xxxxxx
wire                        mul_normal;

assign mul_sign = src0_sign ^ src1_sign;
assign mul_exp = {1'b0, src0_exp_normal} + {1'b0, src1_exp_normal} + BIAS_NEW;
assign mul_man = src0_man_normal * src1_man_normal;
////////////////////////////////////////////////    
// get pre-align shift bits

// right shift bits
wire pre_align_rshift_valid;
wire pre_align_rshift_bits;

priority_encoder #(.WIDTH(2), .LSB_PRIORITY("LOW")) pre_align_r_priority_encoder
(.unencoded(mul_man[2*M_MUL + 1:2*M_MUL]), .valid(pre_align_rshift_valid), .encoded(pre_align_rshift_bits));

// left shift bits
wire [2*(M_MUL + 1) - 2         :0] pre_align_unencoded_left;
wire                                pre_align_lshift_valid;
wire [$clog2(2*M_MUL + 1) - 1   :0] pre_align_lshift_bits;

genvar i;
generate
    for (i = 0; i < 2*M_MUL + 1; i = i + 1) begin: pre_align_reserve_left
        assign pre_align_unencoded_left[i] = mul_man[2*M_MUL - i];
    end
endgenerate

priority_encoder #(.WIDTH(2*M_MUL + 1), .LSB_PRIORITY("HIGH")) pre_align_l_priority_encoder
(.unencoded(pre_align_unencoded_left), .valid(pre_align_lshift_valid), .encoded(pre_align_lshift_bits));

// pre-align shift bits
wire [E_ADD :0] pre_align_bits;

assign pre_align_bits = pre_align_rshift_valid ? {{E_ADD{1'b0}}, pre_align_rshift_bits} : 
                       {{(E_ADD + 1 - $clog2(2*M_MUL + 1)){1'b1}}, ~pre_align_lshift_bits} + 'b1;
////////////////////////////////////////////////    
// pre-align
localparam M_PRE_ALIGN = 2*M_MUL + 1 > M_ADD ? 2*M_MUL + 1 : M_ADD; // .xxxxxx_x

wire [M_PRE_ALIGN + 1   :0] pre_align_man_shift;        // xx.xxxxxx_x

generate
    if(2*M_MUL + 1 > M_ADD) begin
        assign pre_align_man_shift = {mul_man, 1'b0};
    end
    else begin
        assign pre_align_man_shift = {mul_man, {(M_ADD - 2*M_MUL){1'b0}}};
    end
endgenerate

wire [M_PRE_ALIGN + 1   :0] pre_align_man_rshifted;     // xx.xxxxxx_x
wire [M_PRE_ALIGN       :0] pre_align_man_lshifted;     //  x.xxxxxx_x
wire [M_PRE_ALIGN       :0] pre_align_man;              //  x.xxxxxx_x
wire [E_ADD + 1         :0] pre_align_exp;

assign pre_align_man_rshifted = pre_align_man_shift >> pre_align_rshift_bits;
assign pre_align_man_lshifted = pre_align_man_shift[M_PRE_ALIGN:0] << pre_align_lshift_bits;
assign pre_align_man = pre_align_rshift_valid ? pre_align_man_rshifted[M_PRE_ALIGN:0] : pre_align_man_lshifted;
assign pre_align_exp = pre_align_rshift_valid | pre_align_lshift_valid ? {mul_exp[E_ADD], mul_exp} + {pre_align_bits[E_ADD], pre_align_bits} : 'b1;
////////////////////////////////////////////////
// truncation
// localparam TRUNC_M_ADD = 2*M_MUL + 1 < M_ADD + 2;
// localparam M_ADD_REAL = TRUNC_M_ADD ? 2*M_MUL + 4 : M_ADD + 3;  // x.xxxxxx_gr

// wire [M_ADD_REAL    :0] mul_man_trunc; // x.xxxxxx_gr
// wire                    mul_man_sticky;

// generate
//     if(2*M_MUL + 1 <= M_ADD + 2) begin
//         assign mul_man_trunc = {pre_align_man, 2'b00};
//         assign mul_man_sticky = 1'b0; 
//     end
//     else if(2*M_MUL + 1 == M_ADD + 2 + 1) begin
//         assign mul_man_trunc = {pre_align_man, 1'b0};
//         assign mul_man_sticky = 1'b0;
//     end
//     else if(2*M_MUL + 1 == M_ADD + 2 + 2) begin
//         assign mul_man_trunc = pre_align_man;
//         assign mul_man_sticky = 1'b0;
//     end
//     else if(2*M_MUL + 1 > M_ADD + 2 + 2) begin
//         assign mul_man_trunc = pre_align_man[2*M_MUL + 1:2*M_MUL + 1 - M_ADD - 2];
//         assign mul_man_sticky = |mul_man[2*M_MUL + 1 - M_ADD - 3:0];
//     end  
// endgenerate
//////////////////////////////////////////////
// truncation       x.xxxxxx_x
wire [M_ADD + 2 :0] trunc_man; // x.xxxxxx_gr
wire                trunc_sticky;

generate
    if(M_PRE_ALIGN > M_ADD + 2) begin
        assign trunc_man = pre_align_man[M_PRE_ALIGN:M_PRE_ALIGN - M_ADD - 2];
        assign trunc_sticky = |pre_align_man[M_PRE_ALIGN - M_ADD - 3:0];
    end
    else begin
        assign trunc_man = {pre_align_man, {(M_ADD + 2 - M_PRE_ALIGN){1'b0}}};
        assign trunc_sticky = 'b0; 
    end
endgenerate
////////////////////////////////////////////////
// compare
wire                exp_compare;
wire [E_ADD + 1 :0] sub1;
wire [E_ADD + 1 :0] sub2;
wire [E_ADD + 1 :0] exp_diff;
wire [E_ADD - 1 :0] exp_temp;
wire [E_ADD + 1 :0] align_bits;

assign exp_compare = $signed(pre_align_exp) < $signed({2'b0, src2_exp_normal});
assign sub1 = exp_compare ? {2'b0, src2_exp_normal} : pre_align_exp;
assign sub2 = exp_compare ? pre_align_exp : {2'b0, src2_exp_normal};
assign exp_diff = sub1 - sub2;
assign exp_temp = exp_compare ? src2_exp_normal : pre_align_exp[E_ADD - 1:0];
assign align_bits = exp_diff;
////////////////////////////////////////////////
//transfer source code to complement
wire [1                 :0] complement_select;
wire [M_ADD + 2         :0] to_be_complement;   //  x.xxxxxx_gr
wire [M_ADD + 3         :0] man_complement;     // xx.xxxxxx_gr
wire [M_ADD + 3         :0] man1_complement;
wire [M_ADD + 3         :0] man2_complement;

assign complement_select[0] = mul_sign == src2_sign;
assign complement_select[1] = (mul_sign != src2_sign) & mul_sign;
assign to_be_complement = mul_sign ? trunc_man : {src2_man_normal, 2'b0};
assign man_complement = {1'b1, ~to_be_complement} + 'b1;
assign man1_complement = complement_select[1] ? man_complement : {1'b0, trunc_man};
assign man2_complement = complement_select[0] | complement_select[1] ? {1'b0, src2_man_normal, 2'b0} : man_complement;
////////////////////////////////////////////////    
// align
wire signed [M_ADD + 3  :0] align_data_shift;    // xx.xxxxxx_gr
wire        [M_ADD + 3  :0] align_data_shifted;  // xx.xxxxxx_gr
wire        [M_ADD + 3  :0] align_adder_src1;
wire        [M_ADD + 3  :0] align_adder_src2;

assign align_data_shift = exp_compare ? man1_complement : man2_complement;
assign align_data_shifted = align_data_shift >>> align_bits;
assign align_adder_src1 = exp_compare ? man2_complement : man1_complement;
assign align_adder_src2 = align_data_shifted;
////////////////////////////////////////////////
// add
wire [M_ADD + 3 :0] man_sum;   // xx.xxxxxx_gr

assign man_sum = align_adder_src1 + align_adder_src2;
////////////////////////////////////////////////
// transfer complement to source code
wire                sign_d0;
wire [M_ADD + 3 :0] man_source; // xx.xxxxxx_gr

assign sign_d0 = mul_sign == src2_sign ? mul_sign : man_sum[M_ADD + 3];
assign man_source = (mul_sign != src2_sign) & man_sum[M_ADD + 3] ? ~man_sum + 'b1 : man_sum;
////////////////////////////////////////////////
// get pre-normalize shift bits

// right shift bits
wire pre_normal_rshift_valid;
wire pre_normal_rshift_bits;

priority_encoder #(.WIDTH(2), .LSB_PRIORITY("LOW")) pre_normal_r_priority_encoder
(.unencoded(man_source[M_ADD + 3:M_ADD + 2]), .valid(pre_normal_rshift_valid), .encoded(pre_normal_rshift_bits));

// left shift bits
wire [M_ADD + 2             :0] pre_normal_unencoded_left;
wire                            pre_normal_lshift_valid;
wire [$clog2(M_ADD + 3) - 1 :0] pre_normal_lshift_bits;
wire [$clog2(M_ADD + 3) - 1 :0] pre_normal_lshift_bits_true;

assign pre_normal_lshift_bits_true = exp_temp - 'b1 > pre_normal_lshift_bits ? pre_normal_lshift_bits : exp_temp - 'b1;

generate
    for (i = 0; i < M_ADD + 3; i = i + 1) begin: pre_normal_reserve_left
        assign pre_normal_unencoded_left[i] = man_source[M_ADD + 2 - i];
    end
endgenerate

priority_encoder #(.WIDTH(M_ADD + 3), .LSB_PRIORITY("HIGH")) pre_normal_l_priority_encoder
(.unencoded(pre_normal_unencoded_left), .valid(pre_normal_lshift_valid), .encoded(pre_normal_lshift_bits));

// exp shift bits
wire [E_ADD - 1 :0] pre_normal_bits;

assign pre_normal_bits = pre_normal_rshift_valid ? {{E_ADD{1'b0}}, pre_normal_rshift_bits} : {{(E_ADD - $clog2(M_ADD + 3)){1'b1}}, ~pre_normal_lshift_bits_true} + 'b1;
////////////////////////////////////////////////
// pre_normalize
wire [M_ADD + 2 :0] pre_normal_man_rshifted;    // xx.xxxxxx_g
wire                pre_normal_man_reserve;
wire                sticky;
wire [M_ADD + 2 :0] pre_normal_man_lshifted;    //  x.xxxxxx_gr
wire [M_ADD + 2 :0] pre_normal_man;             // xx.xxxxxx_g
wire [E_ADD - 1 :0] pre_normal_exp;  

assign {pre_normal_man_rshifted, pre_normal_man_reserve} = man_source >> pre_normal_rshift_bits;
assign sticky = pre_normal_man_reserve | trunc_sticky;
assign pre_normal_man_lshifted = man_source[M_ADD + 2:0] << pre_normal_lshift_bits_true;
assign pre_normal_man = pre_normal_rshift_valid ? pre_normal_man_rshifted : {1'b0, pre_normal_man_lshifted[M_ADD + 2 :1]};
assign pre_normal_exp = exp_temp + pre_normal_bits;
////////////////////////////////////////////////
// round
wire [M_ADD + 1 :0] man_round1;
wire [M_ADD + 1 :0] man_round0;
reg  [M_ADD + 1 :0] man_rounded; // xx.xxxxxx

assign man_round1 = pre_normal_man[M_ADD + 2:1] + 'b1;
assign man_round0 = pre_normal_man[M_ADD + 2:1];

always @(*) begin
    case({pre_normal_man[0], sticky})
        2'b00: begin
            man_rounded = man_round0;
        end
        2'b01: begin
            man_rounded = man_round0;
        end
        2'b10: begin
            man_rounded = pre_normal_man[1] ? man_round1 : man_round0;
        end
        2'b11: begin
            man_rounded = man_round1;
        end
    endcase
end
////////////////////////////////////////////////
// normalize
wire [E_ADD         :0] normal_exp;         //  x_xxx
wire [E_ADD         :0] normal_bits;        //  x_xxx
wire [M_ADD + 1     :0] normal_man;         //  x.xxxxxx

wire [E_ADD - 1     :0] exp_d0;
wire [M_ADD - 1     :0] man_d0;

assign normal_exp = pre_normal_exp + man_rounded[M_ADD + 1];
assign normal_man = man_rounded[M_ADD + 1] ? man_rounded[M_ADD + 1:1] : man_rounded[M_ADD:0];
assign exp_d0 = normal_exp[E_ADD] ? {E_ADD{1'b1}} : 
                normal_exp == 'b1 & man_rounded[M_ADD + 1:M_ADD] == 'b0 ? {E_ADD{1'b0}} : normal_exp[E_ADD - 1:0];
assign man_d0 = normal_man[M_ADD - 1:0];
////////////////////////////////////////////////
// output
assign result = {sign_d0, exp_d0, man_d0};
assign overflow = normal_exp[E_ADD];

endmodule




/*
 * Priority encoder module
 */
module priority_encoder #
(
    parameter WIDTH = 4,
    // LSB priority: "LOW", "HIGH"
    parameter LSB_PRIORITY = "LOW"
)
(
    input   [WIDTH-1        :0] unencoded,
    output                      valid,
    output  [$clog2(WIDTH)-1:0] encoded
);

// power-of-two width
localparam HALF_WIDTH = 2 ** $clog2(WIDTH) / 2;

generate
    if (WIDTH == 1) begin: priority_encoder_1
        assign valid = unencoded;
        assign encoded = 0;
    end 
    else if (WIDTH == 2) begin: priority_encoder_2
        assign valid = |unencoded;
        if (LSB_PRIORITY == "LOW") begin: priority_encoder_2_low
            assign encoded = unencoded[1];
        end 
        else begin: priority_encoder_2_high
            assign encoded = ~unencoded[0];
        end
    end 
    else begin: priority_encoder_n

        wire [$clog2(HALF_WIDTH)-1:0] encoded_low, encoded_high;
        wire valid_low, valid_high;

        priority_encoder #(
            .WIDTH(HALF_WIDTH),
            .LSB_PRIORITY(LSB_PRIORITY)
        )
        priority_encoder_low (
            .unencoded(unencoded[HALF_WIDTH-1:0]),
            .valid(valid_low),
            .encoded(encoded_low)
        );

        priority_encoder #(
            .WIDTH(HALF_WIDTH),
            .LSB_PRIORITY(LSB_PRIORITY)
        )
        priority_encoder_high (
            .unencoded({{(2 * HALF_WIDTH - WIDTH){1'b0}}, unencoded[WIDTH-1:HALF_WIDTH]}),
            .valid(valid_high),
            .encoded(encoded_high)
        );

        assign valid = valid_low | valid_high;

        if (LSB_PRIORITY == "LOW") begin: priority_encoder_n_low
            assign encoded = valid_high ? {1'b1, encoded_high} : {1'b0, encoded_low};
        end 
        else begin: priority_encoder_n_high
            assign encoded = valid_low ? {1'b0, encoded_low} : {1'b1, encoded_high};
        end
    end
endgenerate

endmodule
