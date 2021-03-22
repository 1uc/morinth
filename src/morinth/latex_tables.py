import os
import itertools
from morinth.math_tools import convergence_rate
from morinth.io import ensure_directory_exists

class LatexConvergenceTable(object):
    """Format convergence data as LaTeX `tabular`."""

    def __init__(self, all_errors, all_rates, resolutions, all_labels):
        self.table = "".join([self.header(all_labels),
                              self.content(all_errors, all_rates, resolutions),
                              self.footer()])

    def write(self, filename):
        ensure_directory_exists(filename=filename)

        with open(filename, "w+") as f:
            f.write(self.table)

    def header(self, all_labels):
        n = len(all_labels)

        header = "".join(
            ["\\begin{{tabular}}{{r\n",
             n*"                S[table-format=3.2e2]r\n",
             "}}\n",
             "\\toprule\n",
             "N & ",
             " & ".join(n*["\n\\multicolumn{{2}}{{c}}{{{:s}}}"]),
             " \\\\\n",
             " & ",
             " & ".join(n*["\n\\multicolumn{{1}}{{c}}{{err}} & \\multicolumn{{1}}{{c}}{{rate}}"]),
             " \\\\\n",
             "\\midrule\n"])
        header = header.format(*all_labels)

        return header

    def footer(self):
        return "".join(["\\bottomrule\n", "\\end{tabular}"])

    def content(self, all_errors, all_rates, resolutions):
        content = self.first_line(resolutions[0], self.extract_line(all_errors, 0))
        for k in range(1, resolutions.size):
            content += self.line(resolutions[k],
                                 self.extract_line(all_errors, k),
                                 self.extract_line(all_rates, k-1))

        return content

    def first_line(self, N, errors):
        data = [N] + errors

        n = len(errors)
        line_pattern = '{:3d} ' + n*"& {: 8.2e}  &      --  " + ' \\\\\n'
        return line_pattern.format(*data)

    def line(self, N, errors, rates):
        data = [N] + list(itertools.chain.from_iterable(zip(errors, rates)))

        n = len(errors)
        line_pattern = '{:3d} ' + n*"& {: 8.2e}  & {: 8.2f} " + ' \\\\\n'
        return line_pattern.format(*data)

    def extract_line(self, data, k):
        return [x[k] for x in data]
