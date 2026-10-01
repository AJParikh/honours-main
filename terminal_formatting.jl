module tf

const TOTAL_WIDTH = 87

function TerminalNewRun(text)
    text_width = length(text)
    padding = max((TOTAL_WIDTH - text_width - 6) ÷ 2, 0)
    println()
    println("\e[1;32m" * "-"^padding * "   $text   " * "-"^padding * "\e[0m")
    println()
end

function TerminalHeading1(text)
    text_width = length(text)
    padding = max(((TOTAL_WIDTH ÷ 2) - text_width - 6) ÷ 2, 0)
    println("\e[1;31m" * "-"^padding * "   $text   " * "-"^padding * "\e[0m")
    println()
end

function TerminalHeading2(text)
    text_width = length(text)
    padding = max(((TOTAL_WIDTH ÷ 2) - text_width - 6) ÷ 2, 0)
    padding = 0
    println("\e[33m" * "-"^padding * "   $text   " * "-"^padding * "\e[0m")
    println()
end

end # module tf
