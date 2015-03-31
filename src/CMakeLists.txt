set(
  MAIN_SOURCES
  "grayworld.cpp"
  "util.cpp"
)

add_library(main ${MAIN_SOURCES})

target_link_libraries(main)
