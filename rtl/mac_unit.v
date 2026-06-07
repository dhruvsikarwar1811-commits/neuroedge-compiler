// NeuroEdge — MAC Unit (Multiply-Accumulate)
// Author: Yuvraj Sikarwar
// Core hardware block for neural network inference

module mac_unit #(
    parameter DATA_WIDTH = 8,
    parameter ACC_WIDTH  = 24
)(
    input  wire                  clk,
    input  wire                  rst_n,
    input  wire                  valid_in,
    input  wire [DATA_WIDTH-1:0] activation,
    input  wire [DATA_WIDTH-1:0] weight,
    input  wire                  acc_clear,
    output reg  [ACC_WIDTH-1:0]  acc_out,
    output reg                   valid_out
);
    wire signed [2*DATA_WIDTH-1:0] product;
    assign product = $signed(activation) * $signed(weight);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            acc_out   <= 0;
            valid_out <= 0;
        end else if (acc_clear) begin
            acc_out   <= {{(ACC_WIDTH-2*DATA_WIDTH){product[2*DATA_WIDTH-1]}}, product};
            valid_out <= valid_in;
        end else if (valid_in) begin
            acc_out   <= acc_out + {{(ACC_WIDTH-2*DATA_WIDTH){product[2*DATA_WIDTH-1]}}, product};
            valid_out <= 1;
        end else begin
            valid_out <= 0;
        end
    end
endmodule
