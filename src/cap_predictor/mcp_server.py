"""CAP Knowledge Base MCP server (stdio) for Cursor chat.

Run:
  python -m cap_predictor.mcp_server

Cursor MCP config (example .cursor/mcp.json):
  {
    \"mcpServers\": {
      \"cap-kb\": {
        \"command\": \"E:/Sushant/LLMCHATBOT_v1_F/Normal Cap Predictor/.venv/Scripts/python.exe\",
        \"args\": [\"-m\", \"cap_predictor.mcp_server\"],
        \"cwd\": \"E:/Sushant/LLMCHATBOT_v1_F/Normal Cap Predictor\",
        \"env\": { \"PYTHONPATH\": \"src\" }
      }
    }
  }

Never invent cutoffs/seats — always call tools.
"""

from __future__ import annotations

import json
from typing import Any

from mcp.server.mcpserver import MCPServer

from cap_predictor.kb.queries import (
    query_cutoffs,
    query_seats,
    query_seats_with_cutoffs,
    search_rulebook,
)

INSTRUCTIONS = (
    "Maharashtra CAP knowledge tools over parsed cutoff PDFs, seat matrix, and "
    "CET Information Brochure. ALWAYS call a tool before answering numerical or "
    "college-list questions. NEVER invent institute codes, cutoffs, seats, or ranks. "
    "closing_percentile is a 0-100 score; closing_merit is selection rank (higher = lower preference)."
)

mcp = MCPServer(
    name="cap-kb",
    title="CAP Knowledge Base",
    instructions=INSTRUCTIONS,
)


def _dump(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str)


@mcp.tool(
    name="query_cutoffs",
    description=(
        "Search CAP cutoff rows. Filters: max/min closing_percentile, min/max closing_merit "
        "(selection rank), category_code (e.g. LNTD, GOPENS), branch_name, year, round_num, "
        "institute_code, university_contains. Returns historical closings only."
    ),
)
def tool_query_cutoffs(
    max_closing_percentile: float | None = None,
    min_closing_percentile: float | None = None,
    min_closing_merit: int | None = None,
    max_closing_merit: int | None = None,
    category_code: str | None = None,
    branch_name: str | None = None,
    year: int | None = None,
    round_num: int | None = None,
    institute_code: int | None = None,
    university_contains: str | None = None,
    limit: int = 100,
) -> str:
    return _dump(
        query_cutoffs(
            max_closing_percentile=max_closing_percentile,
            min_closing_percentile=min_closing_percentile,
            min_closing_merit=min_closing_merit,
            max_closing_merit=max_closing_merit,
            category_code=category_code,
            branch_name=branch_name,
            year=year,
            round_num=round_num,
            institute_code=institute_code,
            university_contains=university_contains,
            limit=limit,
        )
    )


@mcp.tool(
    name="query_seats",
    description=(
        "Search seat-matrix inventory. category_code may be CAP code (LNTD) or matrix "
        "OPEN/SC/…; optional gender G/L, quota HU/OHU/STATE, course, institute, year, min_seats."
    ),
)
def tool_query_seats(
    category_code: str | None = None,
    gender: str | None = None,
    quota: str | None = None,
    course_code: str | None = None,
    course_name_contains: str | None = None,
    institute_code: int | None = None,
    year: int | None = None,
    min_seats: int | None = None,
    limit: int = 100,
) -> str:
    return _dump(
        query_seats(
            category_code=category_code,
            gender=gender,
            quota=quota,
            course_code=course_code,
            course_name_contains=course_name_contains,
            institute_code=institute_code,
            year=year,
            min_seats=min_seats,
            limit=limit,
        )
    )


@mcp.tool(
    name="query_seats_with_cutoffs",
    description=(
        "Find colleges with seats for a CAP category (e.g. LNTD) and attach matching cutoff rows."
    ),
)
def tool_query_seats_with_cutoffs(
    category_code: str,
    year: int | None = None,
    branch_name: str | None = None,
    min_seats: int | None = 1,
    limit: int = 50,
) -> str:
    return _dump(
        query_seats_with_cutoffs(
            category_code=category_code,
            year=year,
            branch_name=branch_name,
            min_seats=min_seats,
            limit=limit,
        )
    )


@mcp.tool(
    name="search_rulebook",
    description="Search CET Information Brochure text for official CAP/eligibility rules.",
)
def tool_search_rulebook(query: str, max_excerpts: int = 5) -> str:
    return _dump(search_rulebook(query, max_excerpts=max_excerpts))


def main() -> None:
    import asyncio

    asyncio.run(mcp.run_stdio_async())


if __name__ == "__main__":
    main()
