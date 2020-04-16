include('church.ax').

thf(
        problem,
        conjecture,
        ?[
                N : ($i > $i) > $i > $i
        ]:
        (
                (mult @ N @ two)
                =
                (plus @ N @ two)
        )
).
