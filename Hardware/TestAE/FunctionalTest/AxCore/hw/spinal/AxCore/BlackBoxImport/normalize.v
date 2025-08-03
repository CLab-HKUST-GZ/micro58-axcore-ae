

/*
 * Normalize
 */
module normalize
#(
    parameter E = 5,
    parameter M = 10,
    parameter INTEGER = 4,
    parameter FRACTION = 20,
    parameter PWIDTH = INTEGER + FRACTION
)
(
    input                   clk   ,

    input   [E + PWIDTH :0] src     ,   //input at d0
    output  [E + M      :0] result      //output at d1
);
////////////////////////////////////////////////
wire                src_sign;
wire [E - 1     :0] src_exp;
wire [PWIDTH - 1:0] src_man;

assign src_sign = src[E + PWIDTH];
assign src_exp  = src[E + PWIDTH - 1:PWIDTH];
assign src_man  = src[PWIDTH - 1:0];
////////////////////////////////////////////////
// transfer complement to source code
wire                    sign_d0;
wire    [PWIDTH - 1:0]  man_temp;

assign sign_d0 = src_sign;
assign man_temp = src_sign ? ~src_man + 'b1 : src_man;
////////////////////////////////////////////////
// get the shift bits

// right shift bits
wire rshift_valid;
wire [$clog2(INTEGER) - 1:0] rshift_bits;

priority_encoder #(.WIDTH(INTEGER), .LSB_PRIORITY("LOW")) r_priority_encoder
(.unencoded(man_temp[PWIDTH - 1:FRACTION]), .valid(rshift_valid), .encoded(rshift_bits));

// left shift bits
wire    [FRACTION                   :0] unencoded_left;
wire                                    lshift_valid;
wire    [$clog2(FRACTION + 1) - 1   :0] lshift_bits;

genvar i;
generate
    for (i = 0; i < FRACTION + 1; i = i + 1) begin: reserve_left
        assign unencoded_left[i] = man_temp[FRACTION-i];
    end
endgenerate

priority_encoder #(.WIDTH(FRACTION + 1), .LSB_PRIORITY("HIGH")) l_priority_encoder
(.unencoded(unencoded_left), .valid(lshift_valid), .encoded(lshift_bits));

// exp shift bits
wire [E - 1 :0] exp_shift_num;

assign exp_shift_num = rshift_valid ? {{(E-$clog2(INTEGER)){1'b0}}, rshift_bits} : {{(E-$clog2(FRACTION + 1)){1'b1}}, ~lshift_bits} + 'b1;
////////////////////////////////////////////////
// normalize
wire [PWIDTH - 1:0] man_rshifted;
wire [PWIDTH - 1:0] man_lshifted;
wire [M + 3     :0] man_normalized;

assign man_rshifted = man_temp >> rshift_bits;
assign man_lshifted = man_temp << lshift_bits;
assign man_normalized = rshift_valid ? man_rshifted[FRACTION + 1:FRACTION - M - 2] : man_lshifted[FRACTION + 1:FRACTION - M - 2];
////////////////////////////////////////////////
// round
wire    [M + 1  :0] man_round1;
wire    [M + 1  :0] man_round0;
reg     [M + 1  :0] man_rounded;

assign man_round1 = man_normalized[M + 3:2] + 'b1;
assign man_round0 = man_normalized[M + 3:2];

always @(*) begin
    case(man_normalized[1:0])
        2'b00: begin
            man_rounded = man_round0;
        end
        2'b01: begin
            man_rounded = man_round0;
        end
        2'b10: begin
            man_rounded = man_normalized[2] ? man_round1 : man_round0;
        end
        2'b11: begin
            man_rounded = man_round1;
        end
    endcase
end
////////////////////////////////////////////////
// normalize
wire [E - 1:0] exp_d0;
wire [M - 1:0] man_d0;

assign exp_d0 = src_exp + exp_shift_num + man_rounded[M + 1];
assign man_d0 = man_rounded[M + 1] ? man_rounded[M:1] : man_rounded[M - 1:0];
////////////////////////////////////////////////
// dff #(M+E+1) result_dff(.clk(clk), .d({sign_d0, exp_d0, man_d0}), .q(result));

reg  [E + M:0] result_reg;

always@(posedge clk) begin                      
    result_reg <= {sign_d0, exp_d0, man_d0};
end

assign result = result_reg;

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

