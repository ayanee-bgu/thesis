# Task

[Domain: algebra | Lead specialist: reasoner]

Solve this problem. State your final answer as a plain number or expression (no LaTeX boxed notation needed — just the value).

Kite $ABCD$ (a quadrilateral with two pairs of adjacent equal sides) has coordinates $A\ (0,7),\ B\ (1,0),\ C\ (12,-2),$ and $D\ (7,8).$ What is the area of $ABCD,$ given that the area of a kite is equal to half the product of its diagonals?

[asy]
string sp(pair P, string P2){return "$" + P2 + "\,(" + string(P.x) + "," + string(P.y) + ")$";}
size(150); defaultpen(fontsize(10)); draw((-5,0)--(15,0),Arrows(4)); draw((0,-5)--(0,10),Arrows(4)); pair A=(0,7),B=(1,0),C=(12,-2),D=(7,8); draw(A--B--C--D--cycle, linewidth(0.7)); draw(A--C, dashed); draw(B--D, dashed); label(sp(A,"A"),A,W); label(sp(B,"B"),B,S); label(sp(C,"C"),C,E); label(sp(D,"D"),D,N);
[/asy]
