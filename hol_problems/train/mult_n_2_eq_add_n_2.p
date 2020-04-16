include('church.ax').

thf(
        problem,
        conjecture,
        ?[
                N : ($i > $i) > $i > $i
        ]:
        (
                (mult @ N @ N)
                =
                (plus @ N @ two)
        )
).
