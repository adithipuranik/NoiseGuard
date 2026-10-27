    )

    trend_df["Monitoring Sample"] = range(

        1,

        len(trend_df) + 1

    )

    st.line_chart(

        trend_df.set_index(

            "Monitoring Sample"

        )["Activity Score"]

    )

    st.caption(

        "Activity scores are relative: "

        "1 = Low, 2 = Moderate, 3 = High. "

        "They do not represent calibrated dB values."

    )

    # ---------------------------------------------

    # MONITORING SUMMARY

    # ---------------------------------------------

    st.subheader(

        " Monitoring Summary"

    )

    average_active = history_df[

        "Active Portion (%)"

    ].mean()

    st.write(

        f"NoiseGuard analyzed **{total_recordings} "

        f"monitoring sample(s)** covering approximately "

        f"**{total_duration:.1f} seconds** of audio."

    )

    st.write(

        f"The average relative active acoustic portion "

        f"across the monitoring samples was "

        f"**{average_active:.2f}%**."

    )

    st.write(

        f"The most frequently detected source was "

        f"**{most_common_source}**."

    )

    if high_events > 0:

        st.warning(

            f"{high_events} high relative acoustic activity "

            f"event(s) were detected. Additional monitoring "

            f"can help determine whether these events are "

            f"persistent."

        )

    else:

        st.success(

            "No high relative acoustic activity events "

            "have been recorded in the current monitoring session."

        )

# =========================================================
